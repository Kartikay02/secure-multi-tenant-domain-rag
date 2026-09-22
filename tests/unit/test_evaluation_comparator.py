"""Unit tests for RunComparator regression detection."""

from app.rag.evaluation.comparator import RunComparator
from app.rag.evaluation.domain import (
    AggregateMetricStats,
    EvaluationRun,
    EvaluationSampleResult,
    LatencySummary,
    MetricCategory,
    MetricResult,
)


def _make_run(
    run_id: str,
    name: str,
    recall: float,
    groundedness: float,
    mean_latency: float,
    failure_rate: float,
    sample_scores: list[tuple[str, float, float]],
) -> EvaluationRun:
    sample_results = [
        EvaluationSampleResult(
            sample_id=s_id,
            question=f"Question {s_id}",
            answer=f"Answer {s_id}",
            metrics={
                "recall_at_5": MetricResult(
                    metric_name="recall_at_5",
                    category=MetricCategory.AUTOMATED,
                    score=r_score,
                ),
                "groundedness": MetricResult(
                    metric_name="groundedness",
                    category=MetricCategory.AUTOMATED,
                    score=g_score,
                ),
            },
        )
        for s_id, r_score, g_score in sample_scores
    ]

    return EvaluationRun(
        run_id=run_id,
        run_name=name,
        dataset_name="bench_v1",
        dataset_version="1.0.0",
        sample_count=len(sample_results),
        success_count=len(sample_results),
        failure_count=0,
        failure_rate=failure_rate,
        aggregate_metrics={
            "recall_at_5": AggregateMetricStats(
                metric_name="recall_at_5",
                category=MetricCategory.AUTOMATED,
                mean=recall,
                median=recall,
                min=0.5,
                max=1.0,
                p95=1.0,
                std_dev=0.1,
                sample_count=len(sample_results),
            ),
            "groundedness": AggregateMetricStats(
                metric_name="groundedness",
                category=MetricCategory.AUTOMATED,
                mean=groundedness,
                median=groundedness,
                min=0.5,
                max=1.0,
                p95=1.0,
                std_dev=0.1,
                sample_count=len(sample_results),
            ),
        },
        latency_summary=LatencySummary(
            mean_ms=mean_latency,
            median_ms=mean_latency,
            p50_ms=mean_latency,
            p90_ms=mean_latency + 10.0,
            p95_ms=mean_latency + 15.0,
            p99_ms=mean_latency + 20.0,
            min_ms=mean_latency - 10.0,
            max_ms=mean_latency + 25.0,
        ),
        sample_results=sample_results,
    )


def test_run_comparator_metrics_and_regressions() -> None:
    comparator = RunComparator(metric_tolerance=0.005, sample_regression_threshold=0.15)

    baseline = _make_run(
        run_id="run-base",
        name="Baseline",
        recall=0.80,
        groundedness=0.90,
        mean_latency=150.0,
        failure_rate=0.02,
        sample_scores=[
            ("s1", 1.0, 1.0),
            ("s2", 0.8, 0.9),
            ("s3", 0.6, 0.8),
        ],
    )

    candidate = _make_run(
        run_id="run-cand",
        name="Candidate",
        recall=0.88,  # +0.08 (IMPROVED)
        groundedness=0.82,  # -0.08 (REGRESSED)
        mean_latency=120.0,  # -30ms
        failure_rate=0.00,  # -0.02
        sample_scores=[
            ("s1", 1.0, 1.0),
            ("s2", 1.0, 0.9),  # recall improved +0.20
            ("s3", 0.65, 0.55),  # groundedness regressed -0.25
        ],
    )

    result = comparator.compare(baseline, candidate)

    # 1. Metric deltas
    assert result.baseline_run_id == "run-base"
    assert result.candidate_run_id == "run-cand"

    recall_delta = result.metric_deltas["recall_at_5"]
    assert recall_delta.delta == 0.08
    assert recall_delta.status == "IMPROVED"
    assert recall_delta.percent_change == 10.0

    ground_delta = result.metric_deltas["groundedness"]
    assert ground_delta.delta == -0.08
    assert ground_delta.status == "REGRESSED"

    # 2. Latency delta
    assert result.mean_latency_delta_ms == -30.0

    # 3. Failure rate delta
    assert result.failure_rate_delta == -0.02

    # 4. Sample-level regression & improvement detection
    assert len(result.regressions) >= 1
    reg = result.regressions[0]
    assert reg.sample_id == "s3"
    assert reg.metric_name == "groundedness"
    assert reg.delta == -0.25

    assert len(result.improvements) >= 1
    imp = result.improvements[0]
    assert imp.sample_id == "s2"
    assert imp.metric_name == "recall_at_5"
    assert imp.delta == 0.20
