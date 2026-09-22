"""Interfaces and protocols for the RAG evaluation framework."""

from typing import Any, Protocol, runtime_checkable

from app.rag.evaluation.domain import (
    EvaluationPrediction,
    EvaluationRun,
    EvaluationRunSummary,
    EvaluationSample,
    MetricCategory,
    MetricResult,
)


@runtime_checkable
class EvaluationMetricProtocol(Protocol):
    """Contract for individual evaluation metrics (automated or LLM-judge)."""

    @property
    def name(self) -> str:
        """Unique identifier name of the metric."""
        ...

    @property
    def category(self) -> MetricCategory:
        """Category distinguishing automated metrics from LLM-as-a-judge metrics."""
        ...

    @property
    def limitations(self) -> str:
        """Explicitly documented assumptions, known blind spots, and edge-case limitations."""
        ...

    async def evaluate(
        self,
        sample: EvaluationSample,
        prediction: EvaluationPrediction,
    ) -> MetricResult:
        """Evaluate a sample prediction against ground-truth and context."""
        ...


@runtime_checkable
class EvaluationTargetProtocol(Protocol):
    """Contract for an evaluation target system (e.g. orchestrator, pipeline, or model)."""

    async def predict(
        self,
        question: str,
        sample: EvaluationSample,
    ) -> EvaluationPrediction:
        """Execute query inference on the target system and return structured prediction."""
        ...


@runtime_checkable
class EvaluationStoreProtocol(Protocol):
    """Contract for persisting, retrieving, and listing evaluation runs."""

    async def save_run(self, run: EvaluationRun) -> str:
        """Persist an evaluation run and return its unique run_id."""
        ...

    async def get_run(self, run_id: str) -> EvaluationRun | None:
        """Retrieve a persisted evaluation run by ID, or None if not found."""
        ...

    async def list_runs(self) -> list[EvaluationRunSummary]:
        """List summary descriptors for all persisted runs ordered by creation timestamp."""
        ...

    async def delete_run(self, run_id: str) -> bool:
        """Delete an evaluation run by ID, returning True if deleted."""
        ...


@runtime_checkable
class LLMJudgeProtocol(Protocol):
    """Contract for an LLM-as-a-judge provider executing evaluation prompts."""

    async def judge(
        self,
        prompt: str,
        system_instruction: str | None = None,
    ) -> tuple[float, str, dict[str, Any]]:
        """Evaluate a prompt and return a tuple of (score [0.0, 1.0], reasoning, metadata)."""
        ...
