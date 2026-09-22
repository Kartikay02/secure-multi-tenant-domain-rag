"""Unit test verifying run_eval.py benchmark execution and golden dataset loading."""

from pathlib import Path

import pytest

from app.rag.evaluation.golden_dataset import DOMAIN_GOLDEN_DATASET
from run_eval import build_default_metrics, run_benchmark


@pytest.mark.asyncio
async def test_run_benchmark_offline_execution(tmp_path: Path) -> None:
    """Validate that run_benchmark executes offline and produces valid reports."""
    eval_run = await run_benchmark(
        dataset=DOMAIN_GOLDEN_DATASET,
        live_endpoint=None,
        output_dir=tmp_path,
        verbose=False,
    )

    assert eval_run.sample_count == len(DOMAIN_GOLDEN_DATASET.samples)
    assert eval_run.failure_count == 0
    assert "answer_groundedness" in eval_run.aggregate_metrics
    assert "recall_at_3" in eval_run.aggregate_metrics
    assert "mrr_at_5" in eval_run.aggregate_metrics

    # Check that report files exist
    assert (tmp_path / "latest_evaluation_report.md").exists()
    assert (tmp_path / "latest_evaluation_report.json").exists()


def test_golden_dataset_integrity() -> None:
    """Validate golden dataset completeness, schema, and sample diversity."""
    assert len(DOMAIN_GOLDEN_DATASET.samples) >= 15
    for sample in DOMAIN_GOLDEN_DATASET.samples:
        assert len(sample.question) > 5
        assert sample.expected_answer is not None
        assert "category" in sample.metadata
        assert "difficulty" in sample.metadata


def test_build_default_metrics_count() -> None:
    metrics = build_default_metrics()
    assert len(metrics) == 10
