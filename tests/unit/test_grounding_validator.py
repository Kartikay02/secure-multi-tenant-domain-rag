"""Unit tests for GroundingValidator covering all 5 core required test cases."""

import uuid

import pytest

from app.core.config import GuardrailSettings
from app.core.exceptions import GuardrailViolationError
from app.rag.context.domain import AssembledContext, ContextDocument
from app.rag.validation.interfaces import GroundingValidatorProtocol
from app.rag.validation.validator import GroundingValidator


def _make_context(doc_content: str | None = None) -> AssembledContext:
    if doc_content is None:
        return AssembledContext(
            formatted_context="",
            documents=[],
            citation_map={},
            total_tokens=0,
            total_chunks=0,
            truncated=False,
            dropped_chunks_count=0,
        )

    doc = ContextDocument(
        citation_id=1,
        citation_label="[1]",
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        source_id="ha_manual.md",
        title="HA Manual",
        page_number=3,
        score=0.92,
        content=doc_content,
        token_count=25,
    )

    return AssembledContext(
        formatted_context=f"--- Context Document [1] ---\n{doc_content}\n",
        documents=[doc],
        citation_map={1: doc},
        total_tokens=30,
        total_chunks=1,
        truncated=False,
        dropped_chunks_count=0,
    )


def test_grounding_validator_implements_protocol() -> None:
    validator = GroundingValidator()
    assert isinstance(validator, GroundingValidatorProtocol)


def test_case_1_fully_grounded_answer() -> None:
    """Case 1: Fully grounded answer with accurate facts and valid citations."""
    context = _make_context(
        "PostgreSQL streaming replication continuously ships write-ahead log (WAL) records to standby nodes."
    )
    validator = GroundingValidator()

    answer = "PostgreSQL streaming replication ships WAL records to standby nodes [1]."
    result = validator.validate(query="How does replication work?", answer=answer, context=context)

    assert result.grounded is True
    assert result.confidence_score >= 0.85
    assert result.unsupported_claims == []
    assert result.citation_errors == []
    assert result.fallback_applied is False
    assert result.final_answer == answer
    assert result.metrics["claim_grounding_ratio"] == 1.0


def test_case_2_partially_grounded_answer() -> None:
    """Case 2: Partially grounded answer containing one supported claim and one unsupported assertion."""
    context = _make_context(
        "PostgreSQL streaming replication continuously ships write-ahead log (WAL) records to standby nodes."
    )
    validator = GroundingValidator()

    answer = (
        "PostgreSQL replication ships WAL records to standby nodes [1]. "
        "It also automatically schedules spacecraft orbital maneuvers with telemetry [1]."
    )
    result = validator.validate(query="What does PostgreSQL do?", answer=answer, context=context)

    assert result.grounded is False
    assert result.fallback_applied is True
    assert len(result.unsupported_claims) == 1
    assert "spacecraft orbital maneuvers" in result.unsupported_claims[0]
    # Fallback refusal message substituted
    assert "I cannot answer this question with sufficient confidence" in result.final_answer


def test_case_3_hallucinated_answer() -> None:
    """Case 3: Completely hallucinated answer unsupported by context evidence."""
    context = _make_context(
        "Redis operates as an in-memory key-value cache with optional AOF persistence."
    )
    validator = GroundingValidator()

    answer = "Quantum entanglement produces instantaneous tachyon beams in warp cores [1]."
    result = validator.validate(query="Explain the technology", answer=answer, context=context)

    assert result.grounded is False
    assert result.confidence_score < 0.50
    assert len(result.unsupported_claims) >= 1
    assert result.fallback_applied is True
    assert "I cannot answer this question with sufficient confidence" in result.final_answer


def test_case_4_no_context_answer() -> None:
    """Case 4: Handling when context is empty."""
    empty_context = _make_context(doc_content=None)
    validator = GroundingValidator()

    # Sub-case 4a: Model attempts to answer when no context exists -> flagged as ungrounded
    answer_attempt = "PostgreSQL uses write-ahead logging."
    res_attempt = validator.validate(
        query="Explain WAL", answer=answer_attempt, context=empty_context
    )

    assert res_attempt.grounded is False
    assert res_attempt.confidence_score == 0.0
    assert res_attempt.insufficient_context is True

    # Sub-case 4b: Model correctly declares explicit uncertainty
    uncertain_answer = (
        "I do not have sufficient information in the provided context to answer this question."
    )
    res_uncertain = validator.validate(
        query="Explain WAL", answer=uncertain_answer, context=empty_context
    )

    assert res_uncertain.grounded is True
    assert res_uncertain.confidence_score == 1.0
    assert res_uncertain.insufficient_context is True
    assert res_uncertain.fallback_applied is False
    assert res_uncertain.final_answer == uncertain_answer


def test_case_5_invalid_citations() -> None:
    """Case 5: Answer referencing phantom / nonexistent citations."""
    context = _make_context("Valid document content.")
    validator = GroundingValidator()

    # Cites [99] which does not exist in context
    answer = "This claim references a phantom source [99]."
    result = validator.validate(query="Query", answer=answer, context=context)

    assert result.grounded is False
    assert any("Citation [99] does not exist" in err for err in result.citation_errors)
    assert result.confidence_score <= 0.40
    assert result.fallback_applied is True


def test_missing_citations_detected() -> None:
    """Verify that factual claims lacking citation markers are flagged."""
    context = _make_context("PostgreSQL uses WAL records.")
    validator = GroundingValidator()

    answer = "PostgreSQL write-ahead logging guarantees database transaction durability."
    result = validator.validate(query="Query", answer=answer, context=context)

    assert result.grounded is False
    assert any("Claim lacks citation" in err for err in result.citation_errors)


def test_fallback_mode_annotate() -> None:
    context = _make_context("Database info.")
    settings = GuardrailSettings(fallback_mode="annotate")
    validator = GroundingValidator(settings=settings)

    hallucination = "Unsupported claim about aliens [1]."
    result = validator.validate(query="Query", answer=hallucination, context=context)

    assert result.grounded is False
    assert result.fallback_applied is True
    assert "[GUARDRAIL WARNING:" in result.final_answer
    assert hallucination in result.final_answer


def test_fallback_mode_raise() -> None:
    context = _make_context("Database info.")
    settings = GuardrailSettings(fallback_mode="raise")
    validator = GroundingValidator(settings=settings)

    hallucination = "Unsupported claim [1]."
    with pytest.raises(GuardrailViolationError) as exc_info:
        validator.validate(query="Query", answer=hallucination, context=context)

    assert exc_info.value.status_code == 422
    assert "Grounding validation failed" in str(exc_info.value)


def test_disabled_validator_empty_context_is_ungrounded() -> None:
    """Verify that an empty context cannot be marked grounded even if validator is disabled."""
    empty_context = _make_context(None)
    settings = GuardrailSettings(enabled=False)
    validator = GroundingValidator(settings=settings)

    result = validator.validate(
        query="Explain something",
        answer="Here is an answer that has no evidence.",
        context=empty_context,
    )

    assert result.grounded is False
    assert result.confidence_score == 0.0
    assert result.insufficient_context is True


def test_disabled_validator_nonempty_context() -> None:
    """Verify that a non-empty context returns grounded=True when validator is explicitly disabled."""
    context = _make_context("Some valid content.")
    settings = GuardrailSettings(enabled=False)
    validator = GroundingValidator(settings=settings)

    result = validator.validate(
        query="Explain something",
        answer="Valid answer.",
        context=context,
    )

    assert result.grounded is True
    assert result.confidence_score == 1.0
    assert result.insufficient_context is False
