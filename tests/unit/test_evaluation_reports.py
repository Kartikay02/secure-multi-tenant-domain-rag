"""Unit tests for ReportGenerator Markdown formatting."""

from app.rag.evaluation.comparator import RunComparator
from app.rag.evaluation.domain import (
    AggregateMetricStats,
    EvaluationRun,
    EvaluationSampleResult,
    LatencySummary,
    MetricCategory,
)
from app.rag.evaluation.reports import ReportGenerator


def test_generate_run_report_markdown() -> None:
    run = EvaluationRun(
        run_name="Prod_Benchmark_Run",
        dataset_name="support_docs_v1",
        dataset_version="1.0.0",
        duration_seconds=5.23,
        sample_count=10,
        success_count=9,
        failure_count=1,
        failure_rate=0.10,
        aggregate_metrics={
            "recall_at_5": AggregateMetricStats(
                metric_name="recall_at_5",
                category=MetricCategory.AUTOMATED,
                mean=0.88,
                median=0.90,
                min=0.60,
                max=1.0,
                p95=1.0,
                std_dev=0.12,
                sample_count=10,
                limitations="Recall@K assumes full gold coverage.",
            ),
            "llm_groundedness": AggregateMetricStats(
                metric_name="llm_groundedness",
                category=MetricCategory.LLM_JUDGE,
                mean=0.92,
                median=0.95,
                min=0.70,
                max=1.0,
                p95=1.0,
                std_dev=0.08,
                sample_count=10,
                limitations="LLM judge exhibits self-preference bias.",
            ),
        },
        latency_summary=LatencySummary(
            mean_ms=115.4,
            median_ms=110.0,
            p50_ms=110.0,
            p90_ms=130.0,
            p95_ms=140.0,
            p99_ms=148.0,
            min_ms=90.0,
            max_ms=150.0,
            stage_averages={"retrieval": 35.0, "generation": 80.4},
        ),
        sample_results=[
            EvaluationSampleResult(
                sample_id="failing-01",
                question="Why did the query crash?",
                answer="",
                error="Database connection pool exhausted",
            )
        ],
    )

    report = ReportGenerator.generate_run_report(run)

    assert "# RAG Evaluation Report: Prod_Benchmark_Run" in report
    assert "Failure Count**: 1 (Failure Rate: 10.0%)" in report
    assert "| **recall_at_5** | Automated | 0.880 |" in report
    assert "| **llm_groundedness** | LLM Judge | 0.920 |" in report
    assert "Recall@K assumes full gold coverage." in report
    assert "| **Mean Latency** | 115.40 ms |" in report
    assert "| `retrieval` | 35.00 ms |" in report
    assert "## Failed Queries" in report
    assert "Database connection pool exhausted" in report


def test_generate_comparison_report_markdown() -> None:
    from tests.unit.test_evaluation_comparator import _make_run

    baseline = _make_run(
        run_id="run-base",
        name="Baseline",
        recall=0.75,
        groundedness=0.90,
        mean_latency=160.0,
        failure_rate=0.05,
        sample_scores=[("s1", 0.7, 0.9), ("s2", 0.8, 0.9)],
    )

    candidate = _make_run(
        run_id="run-cand",
        name="Candidate",
        recall=0.85,
        groundedness=0.82,
        mean_latency=130.0,
        failure_rate=0.00,
        sample_scores=[("s1", 0.9, 0.9), ("s2", 0.8, 0.7)],
    )

    comparator = RunComparator(metric_tolerance=0.005, sample_regression_threshold=0.10)
    comparison = comparator.compare(baseline, candidate)

    report = ReportGenerator.generate_comparison_report(comparison)

    assert "# RAG Evaluation Comparison Report" in report
    assert "Mean Latency Delta**: -30.00 ms" in report
    assert "Failure Rate Delta**: -5.0%" in report
    assert "🟢 IMPROVED" in report  # recall improved
    assert "🔴 REGRESSED" in report  # groundedness regressed
    assert "## Detected Sample Regressions" in report
    assert "`s2`" in report
