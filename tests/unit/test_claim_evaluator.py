"""Unit tests for DeterministicClaimEvaluator and MockClaimEvaluator."""

import uuid

from app.rag.context.domain import ContextDocument
from app.rag.validation.evaluator import DeterministicClaimEvaluator, MockClaimEvaluator


def _make_doc(content: str) -> ContextDocument:
    return ContextDocument(
        citation_id=1,
        citation_label="[1]",
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        source_id="manual.md",
        title="Manual",
        page_number=1,
        score=0.9,
        content=content,
        token_count=20,
    )


def test_claim_evaluator_exact_and_high_support() -> None:
    doc = _make_doc(
        "PostgreSQL physical streaming replication ships write-ahead log (WAL) records directly to standby nodes."
    )
    evaluator = DeterministicClaimEvaluator()

    claim = "Streaming replication ships WAL records to standby nodes [1]."
    supported, score, reason = evaluator.evaluate_claim(claim=claim, evidence=[doc])

    assert supported is True
    assert score >= 0.70
    assert reason is None


def test_claim_evaluator_numeric_mismatch_detected() -> None:
    # Evidence specifies 99.9%, but claim alleges 99.99%
    doc = _make_doc("The service maintains 99.9% uptime SLA under typical load conditions.")
    evaluator = DeterministicClaimEvaluator()

    claim = "The service guarantees 99.99% uptime SLA [1]."
    supported, score, reason = evaluator.evaluate_claim(claim=claim, evidence=[doc])

    assert supported is False
    assert score == 0.20
    assert reason is not None
    assert "Numeric claim mismatch" in reason
    assert "99.99%" in reason


def test_claim_evaluator_unsupported_missing_terms() -> None:
    doc = _make_doc("Kafka uses topic partitions and consumer offsets for event streaming.")
    evaluator = DeterministicClaimEvaluator()

    # Claim talks about completely unrelated quantum mechanics
    claim = "Quantum entanglement enables instantaneous subatomic teleportation [1]."
    supported, score, reason = evaluator.evaluate_claim(claim=claim, evidence=[doc])

    assert supported is False
    assert score < 0.60
    assert reason is not None
    assert "Insufficient term overlap" in reason


def test_claim_evaluator_empty_evidence() -> None:
    evaluator = DeterministicClaimEvaluator()
    supported, score, reason = evaluator.evaluate_claim(claim="Any claim", evidence=[])

    assert supported is False
    assert score == 0.0
    assert "No evidence documents provided" in str(reason)


def test_mock_claim_evaluator_keyword_trigger() -> None:
    doc = _make_doc("Sample evidence text.")
    evaluator = MockClaimEvaluator(fail_on_keywords=["hallucinated", "unverified"])

    # Normal claim succeeds
    supported, score, _ = evaluator.evaluate_claim("Normal claim", [doc])
    assert supported is True

    # Claim containing trigger keyword fails
    supported_fail, score_fail, reason = evaluator.evaluate_claim(
        "This is an unverified claim", [doc]
    )
    assert supported_fail is False
    assert "Simulated failure on keyword 'unverified'" in str(reason)
