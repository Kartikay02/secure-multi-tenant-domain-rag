"""Unit tests for validation domain models."""

from app.rag.validation.domain import ClaimValidation, ValidationResult


def test_claim_validation_model() -> None:
    claim_val = ClaimValidation(
        claim="PostgreSQL WAL replay ensures durability.",
        citations=[1],
        supported=True,
        support_score=0.92,
        missing_citations=False,
        invalid_citations=[],
        unsupported_reason=None,
    )

    assert claim_val.claim.startswith("PostgreSQL")
    assert claim_val.citations == [1]
    assert claim_val.supported is True
    assert claim_val.support_score == 0.92
    assert claim_val.invalid_citations == []


def test_validation_result_model() -> None:
    claim_val = ClaimValidation(
        claim="PostgreSQL uses WAL.",
        citations=[1],
        supported=True,
        support_score=1.0,
        missing_citations=False,
        invalid_citations=[],
    )

    res = ValidationResult(
        grounded=True,
        confidence_score=0.95,
        unsupported_claims=[],
        citation_errors=[],
        claims=[claim_val],
        fallback_applied=False,
        final_answer="PostgreSQL uses WAL [1].",
        insufficient_context=False,
        metrics={"claim_grounding_ratio": 1.0},
    )

    assert res.grounded is True
    assert res.confidence_score == 0.95
    assert len(res.claims) == 1
    assert res.fallback_applied is False
    assert res.final_answer == "PostgreSQL uses WAL [1]."
