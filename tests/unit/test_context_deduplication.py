"""Unit tests for ContentDeduplicator."""

import uuid

from app.rag.context.deduplication import ContentDeduplicator
from app.rag.vector.domain import RetrievalResult


def _make_candidate(text: str, score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        text=text,
        score=score,
        metadata={"source": "test"},
    )


def test_deduplicate_exact_duplicates_keeps_highest_score() -> None:
    c1 = _make_candidate(
        "PostgreSQL multi-version concurrency control manages transactions.", score=0.6
    )
    c2 = _make_candidate(
        "PostgreSQL multi-version concurrency control manages transactions.", score=0.9
    )
    c3 = _make_candidate("Different chunk text about Kafka streaming topics.", score=0.7)

    dedup = ContentDeduplicator()
    results = dedup.deduplicate([c1, c2, c3])

    assert len(results) == 2
    # Verify the identical text has score 0.9 (from c2)
    chunk_scores = {r.text: r.score for r in results}
    assert chunk_scores["PostgreSQL multi-version concurrency control manages transactions."] == 0.9


def test_deduplicate_whitespace_insensitive() -> None:
    c1 = _make_candidate("Text with    multiple   spaces\nand newlines.", score=0.5)
    c2 = _make_candidate("Text with multiple spaces and newlines.", score=0.8)

    dedup = ContentDeduplicator()
    results = dedup.deduplicate([c1, c2])

    assert len(results) == 1
    assert results[0].score == 0.8


def test_deduplicate_near_duplicates_suppressed() -> None:
    # High Jaccard similarity between these two sentences
    c1 = _make_candidate(
        "PostgreSQL streaming replication uses WAL records to synchronize secondary standby database servers.",
        score=0.9,
    )
    c2 = _make_candidate(
        "PostgreSQL streaming replication uses WAL records to synchronize standby database servers.",
        score=0.7,
    )
    c3 = _make_candidate(
        "Redis provides in-memory key-value data structures with optional persistence.",
        score=0.8,
    )

    dedup = ContentDeduplicator(default_threshold=0.80)
    results = dedup.deduplicate([c1, c2, c3])

    assert len(results) == 2
    # c1 is kept (higher score), c2 is suppressed as near-duplicate, c3 is distinct
    scores = [r.score for r in results]
    assert 0.9 in scores
    assert 0.8 in scores
    assert 0.7 not in scores


def test_deduplicate_empty_and_threshold_one() -> None:
    dedup = ContentDeduplicator(default_threshold=1.0)
    assert dedup.deduplicate([]) == []

    c1 = _make_candidate("Slightly different A", score=0.9)
    c2 = _make_candidate("Slightly different B", score=0.8)
    results = dedup.deduplicate([c1, c2], threshold=1.0)
    assert len(results) == 2
