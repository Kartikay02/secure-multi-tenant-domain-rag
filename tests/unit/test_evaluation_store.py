"""Unit tests for InMemoryEvaluationStore and FileEvaluationStore."""

from pathlib import Path

import pytest

from app.rag.evaluation.domain import (
    AggregateMetricStats,
    EvaluationRun,
    LatencySummary,
    MetricCategory,
)
from app.rag.evaluation.store import FileEvaluationStore, InMemoryEvaluationStore


def _create_sample_run(run_id: str = "run-001", name: str = "Test Run") -> EvaluationRun:
    return EvaluationRun(
        run_id=run_id,
        run_name=name,
        dataset_name="benchmark_v1",
        dataset_version="1.0.0",
        sample_count=5,
        success_count=5,
        failure_count=0,
        failure_rate=0.0,
        aggregate_metrics={
            "recall_at_5": AggregateMetricStats(
                metric_name="recall_at_5",
                category=MetricCategory.AUTOMATED,
                mean=0.9,
                median=0.9,
                min=0.8,
                max=1.0,
                p95=1.0,
                std_dev=0.07,
                sample_count=5,
            )
        },
        latency_summary=LatencySummary(
            mean_ms=105.0,
            median_ms=100.0,
            p50_ms=100.0,
            p90_ms=120.0,
            p95_ms=125.0,
            p99_ms=129.0,
            min_ms=90.0,
            max_ms=130.0,
        ),
        sample_results=[],
    )


@pytest.mark.asyncio
async def test_in_memory_evaluation_store() -> None:
    store = InMemoryEvaluationStore()
    run = _create_sample_run()

    # Save
    saved_id = await store.save_run(run)
    assert saved_id == run.run_id

    # Get
    retrieved = await store.get_run(run.run_id)
    assert retrieved is not None
    assert retrieved.run_id == run.run_id
    assert retrieved.run_name == "Test Run"

    # List
    summaries = await store.list_runs()
    assert len(summaries) == 1
    assert summaries[0].run_id == run.run_id
    assert summaries[0].key_metrics["recall_at_5"] == 0.9

    # Delete
    deleted = await store.delete_run(run.run_id)
    assert deleted is True
    assert await store.get_run(run.run_id) is None
    assert len(await store.list_runs()) == 0


@pytest.mark.asyncio
async def test_file_evaluation_store(tmp_path: Path) -> None:
    store = FileEvaluationStore(base_directory=tmp_path)
    run1 = _create_sample_run("run-file-01", "Run File 1")
    run2 = _create_sample_run("run-file-02", "Run File 2")

    # Save runs
    await store.save_run(run1)
    await store.save_run(run2)

    # Check files created
    assert (tmp_path / "run-file-01.json").exists()
    assert (tmp_path / "run-file-02.json").exists()

    # Get run
    retrieved = await store.get_run("run-file-01")
    assert retrieved is not None
    assert retrieved.run_name == "Run File 1"
    assert retrieved.aggregate_metrics["recall_at_5"].mean == 0.9

    # List runs
    summaries = await store.list_runs()
    assert len(summaries) == 2
    run_ids = {s.run_id for s in summaries}
    assert "run-file-01" in run_ids
    assert "run-file-02" in run_ids

    # Delete run
    deleted = await store.delete_run("run-file-01")
    assert deleted is True
    assert not (tmp_path / "run-file-01.json").exists()
    assert await store.get_run("run-file-01") is None
    assert len(await store.list_runs()) == 1
