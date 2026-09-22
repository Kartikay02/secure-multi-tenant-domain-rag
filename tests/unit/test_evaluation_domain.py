"""Unit tests for RAG evaluation domain models."""

from pathlib import Path

from app.rag.evaluation.domain import (
    AggregateMetricStats,
    EvaluationDataset,
    EvaluationRun,
    EvaluationSample,
    LatencySummary,
    MetricCategory,
    MetricResult,
)


def test_evaluation_sample_defaults() -> None:
    sample = EvaluationSample(question="What is HNSW?")
    assert sample.question == "What is HNSW?"
    assert sample.expected_answer is None
    assert sample.expected_sources == []
    assert sample.metadata == {}
    assert len(sample.sample_id) > 0


def test_evaluation_dataset_serialization(tmp_path: Path) -> None:
    dataset = EvaluationDataset(
        name="rag_benchmark_v1",
        description="Core operational RAG evaluation dataset",
        version="1.2.0",
        samples=[
            EvaluationSample(
                sample_id="s-001",
                question="How does hybrid retrieval work?",
                expected_answer="It fuses dense and lexical retrieval.",
                expected_sources=["doc-01", "chunk-05"],
                metadata={"difficulty": "medium"},
            ),
            EvaluationSample(
                sample_id="s-002",
                question="What is BM25?",
                expected_answer="A probabilistic lexical ranking algorithm.",
                expected_sources=["doc-02"],
            ),
        ],
    )

    # JSON round-trip
    json_str = dataset.to_json()
    loaded = EvaluationDataset.from_json(json_str)
    assert loaded.name == dataset.name
    assert loaded.version == "1.2.0"
    assert len(loaded.samples) == 2
    assert loaded.samples[0].question == "How does hybrid retrieval work?"
    assert loaded.samples[0].expected_sources == ["doc-01", "chunk-05"]

    # File save and load
    file_path = tmp_path / "benchmark.json"
    dataset.save_file(file_path)
    assert file_path.exists()

    loaded_from_disk = EvaluationDataset.load_file(file_path)
    assert loaded_from_disk.name == dataset.name
    assert len(loaded_from_disk.samples) == 2


def test_metric_result_model() -> None:
    res = MetricResult(
        metric_name="recall_at_5",
        category=MetricCategory.AUTOMATED,
        score=0.8,
        details={"hits": 4, "total": 5},
        limitations="Assumes full annotation coverage.",
    )
    assert res.metric_name == "recall_at_5"
    assert res.category == MetricCategory.AUTOMATED
    assert res.score == 0.8
    assert res.limitations == "Assumes full annotation coverage."


def test_evaluation_run_serialization() -> None:
    run = EvaluationRun(
        run_name="release_v1_eval",
        dataset_name="rag_benchmark_v1",
        dataset_version="1.0.0",
        duration_seconds=12.45,
        sample_count=2,
        success_count=2,
        failure_count=0,
        failure_rate=0.0,
        aggregate_metrics={
            "recall_at_5": AggregateMetricStats(
                metric_name="recall_at_5",
                category=MetricCategory.AUTOMATED,
                mean=0.85,
                median=0.85,
                min=0.80,
                max=0.90,
                p95=0.90,
                std_dev=0.05,
                sample_count=2,
                limitations="Recall limitations note",
            )
        },
        latency_summary=LatencySummary(
            mean_ms=120.5,
            median_ms=120.0,
            p50_ms=120.0,
            p90_ms=130.0,
            p95_ms=135.0,
            p99_ms=139.0,
            min_ms=110.0,
            max_ms=140.0,
            stage_averages={"retrieval": 45.0, "generation": 75.5},
        ),
        sample_results=[],
    )

    json_str = run.to_json()
    loaded = EvaluationRun.from_json(json_str)
    assert loaded.run_name == "release_v1_eval"
    assert "recall_at_5" in loaded.aggregate_metrics
    assert loaded.aggregate_metrics["recall_at_5"].mean == 0.85
    assert loaded.latency_summary.mean_ms == 120.5
