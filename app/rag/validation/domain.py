"""Domain models for grounding validation, claim verification, and citation guardrails."""

from pydantic import BaseModel, ConfigDict, Field


class ClaimValidation(BaseModel):
    """Validation outcome for an individual factual claim or sentence."""

    model_config = ConfigDict(frozen=True)

    claim: str = Field(..., description="The substantive statement or proposition evaluated")
    citations: list[int] = Field(
        default_factory=list, description="Citations attached to this claim"
    )
    supported: bool = Field(..., description="True if the claim is substantiated by cited evidence")
    support_score: float = Field(
        ..., ge=0.0, le=1.0, description="Evidentiary support score [0.0, 1.0]"
    )
    missing_citations: bool = Field(
        default=False, description="True if substantive claim lacks any citation identifiers"
    )
    invalid_citations: list[int] = Field(
        default_factory=list, description="Citation IDs that do not exist in retrieved context"
    )
    unsupported_reason: str | None = Field(
        default=None, description="Detailed explanation if claim failed validation"
    )


class ValidationResult(BaseModel):
    """Comprehensive validation report produced by the grounding guardrail."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    grounded: bool = Field(
        ..., description="Overall pass/fail verdict indicating if answer is reliably grounded"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Calibrated confidence score based on grounding, citations, and evidence",
    )
    unsupported_claims: list[str] = Field(
        default_factory=list, description="List of specific claims lacking evidentiary support"
    )
    citation_errors: list[str] = Field(
        default_factory=list,
        description="List of citation errors (e.g. non-existent or missing citations)",
    )
    claims: list[ClaimValidation] = Field(
        default_factory=list, description="Detailed breakdown of each evaluated claim"
    )
    fallback_applied: bool = Field(
        default=False, description="True if conservative fallback was executed"
    )
    final_answer: str = Field(
        ...,
        description="The verified answer to serve (original answer or conservative fallback refusal)",
    )
    insufficient_context: bool = Field(
        default=False, description="True if context lacked sufficient evidence to answer the query"
    )
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Component metrics (claim_ratio, citation_ratio, evidence_quality)",
    )
