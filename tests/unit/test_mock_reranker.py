"""Unit tests for MockReranker cross-encoder simulator."""

import uuid

import pytest

from app.core.exceptions import RerankError, RerankTimeoutError
from app.rag.reranking.interfaces import RerankerProtocol
from app.rag.reranking.mock import MockReranker
from app.rag.vector.domain import RetrievalResult


def _make_candidate(text: str, score: float = 0.5) -> RetrievalResult:
    """Helper to create a RetrievalResult with specified text and score."""
    return RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        text=text,
        score=score,
        dense_score=score,
        sparse_score=score,
        metadata={"source": "test_doc"},
    )


@pytest.mark.asyncio
async def test_mock_reranker_implements_protocol() -> None:
    """Verify MockReranker satisfies RerankerProtocol."""
    reranker = MockReranker()
    assert isinstance(reranker, RerankerProtocol)


@pytest.mark.asyncio
async def test_mock_reranker_rescoring_and_reordering() -> None:
    """Verify MockReranker prioritizes chunk with higher query overlap and exact phrase."""
    cand1 = _make_candidate(
        text="The cat sat on the mat. Animals are calm.",
        score=0.9,  # High initial score but low query relevance
    )
    cand2 = _make_candidate(
        text="PostgreSQL multi-version concurrency control manages transactions with snapshot isolation.",
        score=0.4,  # Lower initial score but exact query relevance
    )
    cand3 = _make_candidate(
        text="Distributed systems require partition tolerance and eventual consistency.",
        score=0.5,
    )

    reranker = MockReranker(model_name="test-cross-encoder")
    results = await reranker.rerank(
        query="PostgreSQL concurrency control snapshot isolation",
        candidates=[cand1, cand2, cand3],
        top_k=2,
    )

    assert len(results) == 2
    # cand2 should be boosted to rank 1 due to high term overlap and phrase match
    assert results[0].text == cand2.text
    assert results[0].rerank_score is not None
    assert results[1].rerank_score is not None
    assert results[0].score == results[0].rerank_score
    assert results[0].rerank_score > results[1].rerank_score
    assert results[0].metadata["reranker"] == "test-cross-encoder"
    assert results[0].metadata["source"] == "test_doc"
    assert results[0].dense_score == cand2.dense_score


@pytest.mark.asyncio
async def test_mock_reranker_empty_candidates() -> None:
    """Verify reranking empty list returns empty list."""
    reranker = MockReranker()
    results = await reranker.rerank(query="anything", candidates=[], top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_mock_reranker_top_k_clipping() -> None:
    """Verify reranker limits results to specified top_k."""
    candidates = [_make_candidate(f"Chunk text sample number {i}") for i in range(10)]
    reranker = MockReranker()
    results = await reranker.rerank(query="sample", candidates=candidates, top_k=3)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_mock_reranker_failure_simulation() -> None:
    """Verify should_fail raises RerankError."""
    candidates = [_make_candidate("Sample text")]
    reranker = MockReranker(should_fail=True)

    with pytest.raises(RerankError) as exc_info:
        await reranker.rerank(query="sample", candidates=candidates, top_k=1)

    assert "Mock reranking failure simulated" in str(exc_info.value)
    assert exc_info.value.status_code == 502


@pytest.mark.asyncio
async def test_mock_reranker_timeout_simulation() -> None:
    """Verify simulate_timeout raises RerankTimeoutError."""
    candidates = [_make_candidate("Sample text")]
    reranker = MockReranker(simulate_timeout=True)

    with pytest.raises(RerankTimeoutError) as exc_info:
        await reranker.rerank(query="sample", candidates=candidates, top_k=1)

    assert exc_info.value.status_code == 504
    assert exc_info.value.details["provider"] == "mock"


@pytest.mark.asyncio
async def test_mock_reranker_delay_simulation() -> None:
    """Verify delay parameter functions without error."""
    candidates = [_make_candidate("Sample text")]
    reranker = MockReranker(delay=0.01)
    results = await reranker.rerank(query="sample", candidates=candidates, top_k=1)
    assert len(results) == 1
