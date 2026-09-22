"""Unit tests for chunk ordering strategies."""

import uuid

from app.rag.context.ordering import (
    DocumentOrderOrdering,
    LostInTheMiddleOrdering,
    RelevanceOrdering,
)
from app.rag.vector.domain import RetrievalResult


def _make_candidate(
    score: float, doc_id: uuid.UUID | None = None, chunk_idx: int = 0
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=doc_id or uuid.uuid4(),
        text=f"Chunk with score {score}",
        score=score,
        metadata={"chunk_index": chunk_idx},
    )


def test_relevance_ordering() -> None:
    c1 = _make_candidate(0.4)
    c2 = _make_candidate(0.9)
    c3 = _make_candidate(0.7)

    strategy = RelevanceOrdering()
    ordered = strategy.order([c1, c2, c3])

    assert [c.score for c in ordered] == [0.9, 0.7, 0.4]


def test_lost_in_the_middle_ordering() -> None:
    # 4 candidates with descending scores: 0.9 (rank 0), 0.8 (rank 1), 0.7 (rank 2), 0.6 (rank 3)
    c0 = _make_candidate(0.9)
    c1 = _make_candidate(0.8)
    c2 = _make_candidate(0.7)
    c3 = _make_candidate(0.6)

    strategy = LostInTheMiddleOrdering()
    ordered = strategy.order([c2, c0, c3, c1])

    # Expect: [c0, c2, c3, c1] -> best at index 0, 2nd best at index -1
    scores = [c.score for c in ordered]
    assert scores == [0.9, 0.7, 0.6, 0.8]
    assert scores[0] == 0.9
    assert scores[-1] == 0.8


def test_lost_in_the_middle_small_list() -> None:
    c0 = _make_candidate(0.9)
    c1 = _make_candidate(0.8)

    strategy = LostInTheMiddleOrdering()
    ordered = strategy.order([c1, c0])
    assert [c.score for c in ordered] == [0.9, 0.8]


def test_document_order_ordering() -> None:
    doc1_id = uuid.uuid4()

    c1 = _make_candidate(0.8, doc_id=doc1_id, chunk_idx=2)
    c2 = _make_candidate(0.9, doc_id=doc1_id, chunk_idx=0)
    c3 = _make_candidate(0.5, doc_id=doc1_id, chunk_idx=1)

    strategy = DocumentOrderOrdering()
    ordered = strategy.order([c1, c2, c3])

    # Should sort by chunk_index: 0, 1, 2
    assert [int(c.metadata["chunk_index"]) for c in ordered] == [0, 1, 2]
