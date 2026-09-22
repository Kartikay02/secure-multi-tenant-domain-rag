"""Protocol contracts for claim evaluation and grounding validation."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from app.rag.context.domain import AssembledContext, ContextDocument
from app.rag.validation.domain import ValidationResult


@runtime_checkable
class ClaimEvaluatorProtocol(Protocol):
    """Protocol for scoring evidentiary entailment between an individual claim and source documents."""

    def evaluate_claim(
        self,
        claim: str,
        evidence: Sequence[ContextDocument],
        min_overlap: float = 0.60,
    ) -> tuple[bool, float, str | None]:
        """Evaluate if an assertion is substantiated by the given evidence documents.

        Args:
            claim: The individual claim statement.
            evidence: Context documents cited for this claim.
            min_overlap: Minimum overlap/entailment threshold for support.

        Returns:
            Tuple of (supported: bool, score: float, unsupported_reason: str | None).
        """
        ...


@runtime_checkable
class GroundingValidatorProtocol(Protocol):
    """Protocol for answer grounding validation and hallucination guardrails."""

    def validate(
        self,
        query: str,
        answer: str,
        context: AssembledContext,
    ) -> ValidationResult:
        """Validate answer grounding against retrieved evidence and return a comprehensive report.

        Args:
            query: The user query.
            answer: Generated LLM answer text.
            context: Assembled context chunks.

        Returns:
            ValidationResult with grounded verdict, confidence score, and fallback if necessary.
        """
        ...
