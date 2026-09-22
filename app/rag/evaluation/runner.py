"""Batch evaluation runner and target adapters decoupled from production request flow."""

import asyncio
import math
import statistics
import time
from datetime import UTC, datetime
from typing import Any

from app.rag.evaluation.domain import (
    AggregateMetricStats,
    EvaluationDataset,
    EvaluationPrediction,
    EvaluationRun,
    EvaluationSample,
    EvaluationSampleResult,
    LatencySummary,
    MetricResult,
)
from app.rag.evaluation.interfaces import (
    EvaluationMetricProtocol,
    EvaluationTargetProtocol,
)
from app.rag.orchestration.domain import RAGResponse
from app.rag.orchestration.interfaces import RAGOrchestratorProtocol


class RAGOrchestratorTarget(EvaluationTargetProtocol):
    """Adapter transforming a RAG orchestrator into an evaluation target."""

    def __init__(self, orchestrator: RAGOrchestratorProtocol) -> None:
        self._orchestrator = orchestrator

    async def predict(
        self,
        question: str,
        sample: EvaluationSample,
    ) -> EvaluationPrediction:
        start_time = time.perf_counter()
        try:
            response: RAGResponse = await self._orchestrator.execute(
                query=question,
                request_id=f"eval-{sample.sample_id[:8]}",
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            retrieved_sources: list[str] = []
            retrieved_chunks: list[dict[str, Any]] = []

            for doc in response.referenced_documents:
                # Add source_id, document_id, chunk_id
                if doc.source_id:
                    retrieved_sources.append(doc.source_id)
                retrieved_sources.append(str(doc.document_id))
                retrieved_sources.append(str(doc.chunk_id))

                retrieved_chunks.append(
                    {
                        "citation_id": doc.citation_id,
                        "source_id": doc.source_id,
                        "document_id": str(doc.document_id),
                        "chunk_id": str(doc.chunk_id),
                        "content": doc.content,
                        "score": doc.score,
                        "title": doc.title,
                    }
                )

            return EvaluationPrediction(
                sample_id=sample.sample_id,
                question=question,
                answer=response.answer,
                retrieved_sources=retrieved_sources,
                retrieved_chunks=retrieved_chunks,
                citations=response.citations,
                latency_ms=response.stage_latencies.total_ms or elapsed_ms,
                stage_latencies=response.stage_latencies.to_dict(),
                grounded=response.grounded,
                insufficient_context=response.insufficient_context,
                fallback_applied=response.fallback_applied,
                error=None,
                metadata={
                    "model": response.model,
                    "retrieved_chunks_count": response.retrieved_chunks_count,
                    "context_tokens": response.context_tokens,
                },
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return EvaluationPrediction(
                sample_id=sample.sample_id,
                question=question,
                answer="",
                retrieved_sources=[],
                retrieved_chunks=[],
                citations=[],
                latency_ms=elapsed_ms,
                stage_latencies={},
                grounded=False,
                insufficient_context=False,
                fallback_applied=False,
                error=str(exc),
                metadata={},
            )


class EvaluationRunner:
    """Orchestrates batch benchmark evaluation across datasets and metrics."""

    def __init__(
        self,
        target: EvaluationTargetProtocol,
        metrics: list[EvaluationMetricProtocol],
        max_concurrency: int = 4,
    ) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be greater than zero")
        self._target = target
        self._metrics = metrics
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def evaluate_sample(self, sample: EvaluationSample) -> EvaluationSampleResult:
        """Evaluate a single benchmark item."""
        async with self._semaphore:
            prediction = await self._target.predict(sample.question, sample)

            metric_tasks = [m.evaluate(sample, prediction) for m in self._metrics]
            metric_results = await asyncio.gather(*metric_tasks, return_exceptions=True)

            metrics_dict: dict[str, MetricResult] = {}
            for metric, res in zip(self._metrics, metric_results, strict=True):
                if isinstance(res, Exception):
                    metrics_dict[metric.name] = MetricResult(
                        metric_name=metric.name,
                        category=metric.category,
                        score=0.0,
                        details={"evaluation_error": str(res)},
                        limitations=metric.limitations,
                    )
                elif isinstance(res, MetricResult):
                    metrics_dict[metric.name] = res

            return EvaluationSampleResult(
                sample_id=sample.sample_id,
                question=sample.question,
                answer=prediction.answer,
                expected_answer=sample.expected_answer,
                expected_sources=sample.expected_sources,
                retrieved_sources=prediction.retrieved_sources,
                metrics=metrics_dict,
                latency_ms=prediction.latency_ms,
                stage_latencies=prediction.stage_latencies,
                error=prediction.error,
            )

    def _compute_percentile(self, values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        k = (len(sorted_vals) - 1) * (percentile / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_vals[int(k)]
        d0 = sorted_vals[int(f)] * (c - k)
        d1 = sorted_vals[int(c)] * (k - f)
        return d0 + d1

    def _calculate_aggregate_metrics(
        self,
        sample_results: list[EvaluationSampleResult],
    ) -> dict[str, AggregateMetricStats]:
        aggregates: dict[str, AggregateMetricStats] = {}

        for metric in self._metrics:
            scores: list[float] = []
            for s in sample_results:
                if metric.name in s.metrics:
                    scores.append(s.metrics[metric.name].score)

            if not scores:
                continue

            mean_val = statistics.mean(scores)
            median_val = statistics.median(scores)
            min_val = min(scores)
            max_val = max(scores)
            p95_val = self._compute_percentile(scores, 95.0)
            std_dev_val = statistics.stdev(scores) if len(scores) > 1 else 0.0

            aggregates[metric.name] = AggregateMetricStats(
                metric_name=metric.name,
                category=metric.category,
                mean=round(mean_val, 4),
                median=round(median_val, 4),
                min=round(min_val, 4),
                max=round(max_val, 4),
                p95=round(p95_val, 4),
                std_dev=round(std_dev_val, 4),
                sample_count=len(scores),
                limitations=metric.limitations,
            )

        return aggregates

    def _calculate_latency_summary(
        self,
        sample_results: list[EvaluationSampleResult],
    ) -> LatencySummary:
        latencies = [s.latency_ms for s in sample_results if s.latency_ms > 0.0]
        if not latencies:
            return LatencySummary(
                mean_ms=0.0,
                median_ms=0.0,
                p50_ms=0.0,
                p90_ms=0.0,
                p95_ms=0.0,
                p99_ms=0.0,
                min_ms=0.0,
                max_ms=0.0,
                stage_averages={},
            )

        stage_totals: dict[str, list[float]] = {}
        for s in sample_results:
            for stage, ms in s.stage_latencies.items():
                stage_totals.setdefault(stage, []).append(ms)

        stage_averages = {
            stage: round(statistics.mean(times), 2)
            for stage, times in stage_totals.items()
            if times
        }

        return LatencySummary(
            mean_ms=round(statistics.mean(latencies), 2),
            median_ms=round(statistics.median(latencies), 2),
            p50_ms=round(self._compute_percentile(latencies, 50.0), 2),
            p90_ms=round(self._compute_percentile(latencies, 90.0), 2),
            p95_ms=round(self._compute_percentile(latencies, 95.0), 2),
            p99_ms=round(self._compute_percentile(latencies, 99.0), 2),
            min_ms=round(min(latencies), 2),
            max_ms=round(max(latencies), 2),
            stage_averages=stage_averages,
        )

    async def run_batch(
        self,
        dataset: EvaluationDataset,
        run_name: str = "eval_run",
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationRun:
        """Run batch evaluation across all samples in the dataset."""
        started_at = datetime.now(UTC)
        start_perf = time.perf_counter()

        tasks = [self.evaluate_sample(s) for s in dataset.samples]
        sample_results = await asyncio.gather(*tasks)

        duration_seconds = time.perf_counter() - start_perf
        completed_at = datetime.now(UTC)

        sample_count = len(sample_results)
        failure_count = sum(1 for s in sample_results if s.error is not None)
        success_count = sample_count - failure_count
        failure_rate = (failure_count / sample_count) if sample_count > 0 else 0.0

        aggregate_metrics = self._calculate_aggregate_metrics(sample_results)
        latency_summary = self._calculate_latency_summary(sample_results)

        return EvaluationRun(
            run_name=run_name,
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=round(duration_seconds, 2),
            sample_count=sample_count,
            success_count=success_count,
            failure_count=failure_count,
            failure_rate=round(failure_rate, 4),
            aggregate_metrics=aggregate_metrics,
            latency_summary=latency_summary,
            sample_results=sample_results,
            metadata=metadata or {},
        )
