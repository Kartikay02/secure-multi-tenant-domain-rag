"""Unit tests for EvaluationRunner and target adapters."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.rag.context.domain import ContextDocument
from app.rag.evaluation.domain import (
    EvaluationDataset,
    EvaluationPrediction,
    EvaluationSample,
)
from app.rag.evaluation.interfaces import (
    EvaluationTargetProtocol,
)
from app.rag.evaluation.metrics.retrieval import RecallAtKMetric
from app.rag.evaluation.runner import EvaluationRunner, RAGOrchestratorTarget
from app.rag.orchestration.domain import PipelineStageLatency, RAGResponse


class MockTarget(EvaluationTargetProtocol):
    """Mock evaluation target returning deterministic predictions or simulated exceptions."""

    def __init__(self, fail_questions: set[str] | None = None) -> None:
        self.fail_questions = fail_questions or set()
        self.call_count = 0

    async def predict(
        self,
        question: str,
        sample: EvaluationSample,
    ) -> EvaluationPrediction:
        self.call_count += 1
        if question in self.fail_questions:
            return EvaluationPrediction(
                sample_id=sample.sample_id,
                question=question,
                error="Simulated target failure: connection timeout",
                latency_ms=50.0,
            )

        return EvaluationPrediction(
            sample_id=sample.sample_id,
            question=question,
            answer=f"Answer for {question} [1].",
            retrieved_sources=["doc_1", "doc_2"],
            retrieved_chunks=[
                {"citation_id": 1, "source_id": "doc_1", "content": f"Content for {question}."}
            ],
            citations=[1],
            latency_ms=100.0,
            stage_latencies={"retrieval_ms": 30.0, "generation_ms": 70.0},
            grounded=True,
        )


@pytest.mark.asyncio
async def test_evaluation_runner_batch_execution() -> None:
    target = MockTarget()
    metric = RecallAtKMetric(k=2)
    runner = EvaluationRunner(target=target, metrics=[metric], max_concurrency=2)

    dataset = EvaluationDataset(
        name="test_dataset",
        samples=[
            EvaluationSample(sample_id="1", question="Q1", expected_sources=["doc_1"]),
            EvaluationSample(sample_id="2", question="Q2", expected_sources=["doc_2"]),
            EvaluationSample(sample_id="3", question="Q3", expected_sources=["doc_3"]),
        ],
    )

    run = await runner.run_batch(dataset, run_name="batch_test")

    assert run.run_name == "batch_test"
    assert run.sample_count == 3
    assert run.success_count == 3
    assert run.failure_count == 0
    assert run.failure_rate == 0.0

    assert "recall_at_2" in run.aggregate_metrics
    stats = run.aggregate_metrics["recall_at_2"]
    # Q1: doc_1 in [doc_1, doc_2] -> 1.0; Q2: doc_2 in [doc_1, doc_2] -> 1.0; Q3: doc_3 not in top 2 -> 0.0
    # Mean: (1.0 + 1.0 + 0.0) / 3 = 0.6667
    assert stats.sample_count == 3
    assert 0.66 <= stats.mean <= 0.67

    assert run.latency_summary.mean_ms == 100.0
    assert run.latency_summary.stage_averages["retrieval_ms"] == 30.0


@pytest.mark.asyncio
async def test_evaluation_runner_failure_rate() -> None:
    # 1 of 4 fails
    target = MockTarget(fail_questions={"Failing Question"})
    runner = EvaluationRunner(target=target, metrics=[RecallAtKMetric(k=2)], max_concurrency=2)

    dataset = EvaluationDataset(
        name="failure_dataset",
        samples=[
            EvaluationSample(sample_id="1", question="Q1", expected_sources=["doc_1"]),
            EvaluationSample(
                sample_id="2", question="Failing Question", expected_sources=["doc_1"]
            ),
            EvaluationSample(sample_id="3", question="Q3", expected_sources=["doc_1"]),
            EvaluationSample(sample_id="4", question="Q4", expected_sources=["doc_1"]),
        ],
    )

    run = await runner.run_batch(dataset)
    assert run.sample_count == 4
    assert run.success_count == 3
    assert run.failure_count == 1
    assert run.failure_rate == 0.25

    failed_sample = next(s for s in run.sample_results if s.sample_id == "2")
    assert failed_sample.error is not None
    assert "Simulated target failure" in failed_sample.error


@pytest.mark.asyncio
async def test_rag_orchestrator_target_adapter() -> None:
    mock_orchestrator = MagicMock()
    doc_id = uuid.uuid4()
    chunk_id = uuid.uuid4()

    mock_doc = ContextDocument(
        citation_id=1,
        citation_label="[1]",
        chunk_id=chunk_id,
        document_id=doc_id,
        source_id="manual.md",
        title="Manual",
        page_number=1,
        score=0.92,
        content="Documentation content for RAG.",
        token_count=15,
    )

    mock_response = RAGResponse(
        query="normalized query",
        raw_query="raw query",
        answer="Grounded answer text [1].",
        grounded=True,
        confidence_score=0.95,
        citations=[1],
        referenced_documents=[mock_doc],
        citation_manifest='[1] "Manual" (Source: manual.md)',
        stage_latencies=PipelineStageLatency(
            retrieval_ms=25.0,
            generation_ms=85.0,
            total_ms=120.0,
        ),
        request_id="req-123",
        model="gpt-4o",
    )
    mock_orchestrator.execute = AsyncMock(return_value=mock_response)

    adapter = RAGOrchestratorTarget(mock_orchestrator)
    sample = EvaluationSample(sample_id="s1", question="raw query")

    prediction = await adapter.predict("raw query", sample)

    assert prediction.sample_id == "s1"
    assert prediction.answer == "Grounded answer text [1]."
    assert prediction.grounded is True
    assert "manual.md" in prediction.retrieved_sources
    assert str(doc_id) in prediction.retrieved_sources
    assert str(chunk_id) in prediction.retrieved_sources
    assert prediction.latency_ms == 120.0
    assert prediction.stage_latencies["retrieval_ms"] == 25.0
    assert prediction.error is None
