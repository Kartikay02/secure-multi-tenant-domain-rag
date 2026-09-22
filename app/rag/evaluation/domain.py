"""Domain models for RAG evaluation subsystem."""

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MetricCategory(StrEnum):
    """Categorization distinguishing deterministic automated metrics from LLM-as-a-judge."""

    AUTOMATED = "automated"
    LLM_JUDGE = "llm_judge"


class EvaluationSample(BaseModel):
    """A single evaluation benchmark item."""

    model_config = ConfigDict(frozen=True)

    sample_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the benchmark item",
    )
    question: str = Field(..., min_length=1, description="Evaluation query or question")
    expected_answer: str | None = Field(
        default=None, description="Ground-truth target answer string"
    )
    expected_sources: list[str] = Field(
        default_factory=list,
        description="List of expected document IDs, source IDs, or chunk identifiers",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary sample metadata (category, difficulty, topic)",
    )


class EvaluationDataset(BaseModel):
    """Benchmark dataset containing evaluation samples and versioning metadata."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., min_length=1, description="Dataset name")
    description: str = Field(default="", description="Dataset description")
    version: str = Field(default="1.0.0", description="Semantic dataset version")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of dataset creation",
    )
    samples: list[EvaluationSample] = Field(
        default_factory=list, description="Ordered collection of benchmark samples"
    )

    def to_json(self, indent: int = 2) -> str:
        """Serialize dataset to JSON formatted string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str) -> "EvaluationDataset":
        """Deserialize dataset from JSON formatted string."""
        return cls.model_validate_json(data)

    def save_file(self, path: Path | str) -> None:
        """Persist dataset to file on disk."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load_file(cls, path: Path | str) -> "EvaluationDataset":
        """Load dataset from file on disk."""
        content = Path(path).read_text(encoding="utf-8")
        return cls.from_json(content)


class EvaluationPrediction(BaseModel):
    """Inference output captured from an evaluation target for a single sample."""

    model_config = ConfigDict(extra="ignore")

    sample_id: str = Field(..., description="Correlated sample identifier")
    question: str = Field(..., description="Query evaluated")
    answer: str = Field(default="", description="Generated answer text")
    retrieved_sources: list[str] = Field(
        default_factory=list,
        description="List of retrieved source/document/chunk identifiers in ranked order",
    )
    retrieved_chunks: list[dict[str, Any]] = Field(
        default_factory=list, description="Raw chunk payloads with scores and content"
    )
    citations: list[int] = Field(
        default_factory=list, description="Citation integer identifiers cited in answer"
    )
    latency_ms: float = Field(default=0.0, ge=0.0, description="End-to-end target latency in ms")
    stage_latencies: dict[str, float] = Field(
        default_factory=dict, description="Per-stage latency breakdown in ms"
    )
    grounded: bool = Field(default=True, description="True if answer passed grounding check")
    insufficient_context: bool = Field(
        default=False, description="True if target refused due to insufficient context"
    )
    fallback_applied: bool = Field(
        default=False, description="True if conservative fallback refusal was triggered"
    )
    error: str | None = Field(
        default=None, description="Exception or error message if query execution failed"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Target diagnostic and pipeline telemetry"
    )


class MetricResult(BaseModel):
    """Result produced by evaluating a single metric on a sample prediction."""

    model_config = ConfigDict(frozen=True)

    metric_name: str = Field(..., description="Identifier name of the metric")
    category: MetricCategory = Field(
        ..., description="Whether this is an automated or LLM-judge metric"
    )
    score: float = Field(..., description="Evaluated score (typically normalized [0.0, 1.0])")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Component diagnostic details for the score"
    )
    limitations: str = Field(
        default="", description="Explicitly documented assumptions, blind spots, and limitations"
    )


class AggregateMetricStats(BaseModel):
    """Statistical summary across all evaluated samples for a specific metric."""

    model_config = ConfigDict(frozen=True)

    metric_name: str
    category: MetricCategory
    mean: float
    median: float
    min: float
    max: float
    p95: float
    std_dev: float
    sample_count: int
    limitations: str = ""


class LatencySummary(BaseModel):
    """Statistical summary of query latencies across an evaluation run."""

    model_config = ConfigDict(frozen=True)

    mean_ms: float
    median_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    stage_averages: dict[str, float] = Field(default_factory=dict)


class EvaluationSampleResult(BaseModel):
    """Combined sample, prediction, and metric results for one benchmark item."""

    model_config = ConfigDict(extra="ignore")

    sample_id: str
    question: str
    answer: str
    expected_answer: str | None = None
    expected_sources: list[str] = Field(default_factory=list)
    retrieved_sources: list[str] = Field(default_factory=list)
    metrics: dict[str, MetricResult] = Field(default_factory=dict)
    latency_ms: float = 0.0
    stage_latencies: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


class EvaluationRun(BaseModel):
    """A complete persisted benchmark evaluation run with full telemetry."""

    model_config = ConfigDict(extra="ignore")

    run_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the evaluation run",
    )
    run_name: str = Field(default="eval_run", description="Human-readable run name")
    dataset_name: str = Field(..., description="Name of benchmark dataset evaluated")
    dataset_version: str = Field(default="1.0.0", description="Version of dataset")
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Run start timestamp"
    )
    completed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Run completion timestamp"
    )
    duration_seconds: float = Field(default=0.0, description="Total run duration in seconds")
    sample_count: int = Field(default=0, ge=0, description="Total number of evaluated samples")
    success_count: int = Field(
        default=0, ge=0, description="Number of successfully evaluated samples"
    )
    failure_count: int = Field(default=0, ge=0, description="Number of failed samples")
    failure_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fraction of samples that failed [0.0, 1.0]"
    )
    aggregate_metrics: dict[str, AggregateMetricStats] = Field(
        default_factory=dict, description="Aggregated metric statistics keyed by metric name"
    )
    latency_summary: LatencySummary
    sample_results: list[EvaluationSampleResult] = Field(
        default_factory=list, description="Per-sample evaluation results"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Target configuration, model versions, and run tags"
    )

    def to_json(self, indent: int = 2) -> str:
        """Serialize evaluation run to JSON."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str) -> "EvaluationRun":
        """Deserialize evaluation run from JSON."""
        return cls.model_validate_json(data)


class EvaluationRunSummary(BaseModel):
    """Compact summary of an evaluation run for fast index listing."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    run_name: str
    dataset_name: str
    dataset_version: str
    created_at: datetime
    sample_count: int
    failure_rate: float
    mean_latency_ms: float
    key_metrics: dict[str, float] = Field(default_factory=dict)


class MetricDelta(BaseModel):
    """Comparison delta between baseline and candidate runs for a single metric."""

    model_config = ConfigDict(frozen=True)

    metric_name: str
    category: MetricCategory
    baseline_mean: float
    candidate_mean: float
    delta: float  # candidate - baseline
    percent_change: float  # ((candidate - baseline) / baseline) * 100
    status: str  # "IMPROVED", "REGRESSED", "UNCHANGED"


class SampleRegression(BaseModel):
    """Sample-level regression where candidate performed significantly worse than baseline."""

    model_config = ConfigDict(frozen=True)

    sample_id: str
    question: str
    metric_name: str
    baseline_score: float
    candidate_score: float
    delta: float


class RunComparisonResult(BaseModel):
    """Comprehensive comparison between a baseline evaluation run and candidate run."""

    model_config = ConfigDict(extra="ignore")

    baseline_run_id: str
    candidate_run_id: str
    dataset_name: str
    metric_deltas: dict[str, MetricDelta] = Field(default_factory=dict)
    mean_latency_delta_ms: float = 0.0
    p95_latency_delta_ms: float = 0.0
    failure_rate_delta: float = 0.0
    regressions: list[SampleRegression] = Field(default_factory=list)
    improvements: list[SampleRegression] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
