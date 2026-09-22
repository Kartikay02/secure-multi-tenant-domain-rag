"""Unit tests for RerankingPipeline, NoOpReranker, and RerankerFactory."""

import uuid
from typing import Any

import pytest
from pydantic import ValidationError

from app.core.config import RetrievalSettings
from app.rag.reranking.cohere import CohereReranker
from app.rag.reranking.factory import RerankerFactory
from app.rag.reranking.interfaces import RerankerProtocol
from app.rag.reranking.mock import MockReranker
from app.rag.reranking.noop import NoOpReranker
from app.rag.reranking.pipeline import RerankingPipeline
from app.rag.retrieval.interfaces import RetrieverProtocol
from app.rag.vector.domain import RetrievalResult


class DummyRetriever(RetrieverProtocol):
    """Simple in-memory test retriever."""

    def __init__(self, candidates: list[RetrievalResult]) -> None:
        self._candidates = candidates
        self.last_top_k: int | None = None
        self.last_query: str | None = None
        self.last_filters: dict[str, Any] | None = None

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        self.last_query = query
        self.last_top_k = top_k
        self.last_filters = filter_metadata
        res = self._candidates[:top_k]
        if score_threshold is not None:
            res = [r for r in res if r.score >= score_threshold]
        return res


def _make_candidate(text: str, score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        text=text,
        score=score,
        dense_score=score,
        sparse_score=score,
        metadata={"source": "unit_test"},
    )


@pytest.mark.asyncio
async def test_reranking_pipeline_normal_flow() -> None:
    """Verify normal two-stage candidate retrieval and mock reranking."""
    cand1 = _make_candidate("Chunk about weather forecasting and rain", score=0.8)
    cand2 = _make_candidate("Chunk about PostgreSQL WAL streaming and transactions", score=0.4)
    cand3 = _make_candidate("Chunk about general computing hardware", score=0.3)

    retriever = DummyRetriever([cand1, cand2, cand3])
    reranker = MockReranker(model_name="test-reranker")
    pipeline = RerankingPipeline(
        retriever=retriever,
        reranker=reranker,
        candidates_k=10,
        top_k=2,
    )

    results = await pipeline.search(query="PostgreSQL WAL transactions")

    assert retriever.last_top_k == 10
    assert len(results) == 2
    # cand2 should be reranked to position 0
    assert results[0].text == cand2.text
    assert results[0].rerank_score is not None
    assert results[0].metadata["reranker"] == "test-reranker"
    assert results[0].metadata["source"] == "unit_test"


@pytest.mark.asyncio
async def test_reranking_pipeline_graceful_fallback_on_error() -> None:
    """CRITICAL: Verify pipeline gracefully falls back to hybrid candidates when reranker fails."""
    cand1 = _make_candidate("Top hybrid result", score=0.9)
    cand2 = _make_candidate("Second hybrid result", score=0.8)
    cand3 = _make_candidate("Third hybrid result", score=0.7)

    retriever = DummyRetriever([cand1, cand2, cand3])
    failing_reranker = MockReranker(should_fail=True)
    pipeline = RerankingPipeline(
        retriever=retriever,
        reranker=failing_reranker,
        candidates_k=10,
        top_k=2,
    )

    # Search MUST NOT raise an exception
    results = await pipeline.search(query="any query")

    assert len(results) == 2
    # Should return top-2 hybrid candidates in original order
    assert results[0].chunk_id == cand1.chunk_id
    assert results[0].score == cand1.score
    assert results[0].rerank_score is None
    assert results[0].metadata["rerank_fallback"] is True
    assert "Mock reranking failure simulated" in results[0].metadata["rerank_error"]

    assert results[1].chunk_id == cand2.chunk_id
    assert results[1].metadata["rerank_fallback"] is True


@pytest.mark.asyncio
async def test_reranking_pipeline_graceful_fallback_on_timeout() -> None:
    """Verify pipeline gracefully falls back on reranker timeout."""
    cand1 = _make_candidate("Top hybrid result", score=0.9)
    retriever = DummyRetriever([cand1])
    timing_out_reranker = MockReranker(simulate_timeout=True)

    pipeline = RerankingPipeline(
        retriever=retriever,
        reranker=timing_out_reranker,
        candidates_k=5,
        top_k=1,
    )

    results = await pipeline.search(query="any query")
    assert len(results) == 1
    assert results[0].metadata["rerank_fallback"] is True
    assert "timed out" in results[0].metadata["rerank_error"]


@pytest.mark.asyncio
async def test_reranking_pipeline_empty_query() -> None:
    """Verify empty query returns empty list immediately."""
    retriever = DummyRetriever([_make_candidate("sample")])
    pipeline = RerankingPipeline(retriever=retriever, reranker=MockReranker())

    results = await pipeline.search(query="   ")
    assert results == []
    assert retriever.last_query is None


@pytest.mark.asyncio
async def test_reranking_pipeline_no_candidates() -> None:
    """Verify empty candidates from retriever returns empty list immediately."""
    retriever = DummyRetriever([])
    pipeline = RerankingPipeline(retriever=retriever, reranker=MockReranker())

    results = await pipeline.search(query="query with no results")
    assert results == []


@pytest.mark.asyncio
async def test_reranking_pipeline_retrieve_conformance() -> None:
    """Verify RerankingPipeline satisfies RetrieverProtocol with score_threshold."""
    cand1 = _make_candidate("Chunk with high score", score=0.85)
    cand2 = _make_candidate("Chunk with lower score", score=0.20)

    retriever = DummyRetriever([cand1, cand2])
    pipeline = RerankingPipeline(retriever=retriever, reranker=NoOpReranker())

    # Check protocol check
    assert isinstance(pipeline, RetrieverProtocol)

    results = await pipeline.retrieve(
        query="test",
        top_k=5,
        score_threshold=0.5,
    )
    assert len(results) == 1
    assert results[0].text == cand1.text


@pytest.mark.asyncio
async def test_noop_reranker() -> None:
    """Verify NoOpReranker passes candidate slices through unmodified."""
    cand1 = _make_candidate("chunk 1", score=0.9)
    cand2 = _make_candidate("chunk 2", score=0.7)
    cand3 = _make_candidate("chunk 3", score=0.5)

    noop = NoOpReranker()
    assert isinstance(noop, RerankerProtocol)

    results = await noop.rerank(query="irrelevant", candidates=[cand1, cand2, cand3], top_k=2)
    assert len(results) == 2
    assert results[0] == cand1
    assert results[1] == cand2


def test_reranker_factory() -> None:
    """Verify RerankerFactory resolves appropriate provider based on settings."""
    # 1. Disabled
    cfg_disabled = RetrievalSettings(reranker_enabled=False)
    assert isinstance(RerankerFactory.create_reranker(cfg_disabled), NoOpReranker)

    # 2. Provider none
    cfg_none = RetrievalSettings(reranker_enabled=True, reranker_provider="none")
    assert isinstance(RerankerFactory.create_reranker(cfg_none), NoOpReranker)

    # 3. Provider mock
    cfg_mock = RetrievalSettings(reranker_enabled=True, reranker_provider="mock")
    reranker_mock = RerankerFactory.create_reranker(cfg_mock)
    assert isinstance(reranker_mock, MockReranker)

    # 4. Provider cohere
    cfg_cohere = RetrievalSettings(
        reranker_enabled=True,
        reranker_provider="cohere",
        reranker_api_key="coh-test-key",
    )
    reranker_cohere = RerankerFactory.create_reranker(cfg_cohere)
    assert isinstance(reranker_cohere, CohereReranker)

    # 5. Provider flashrank (unsupported yet)
    cfg_flash = RetrievalSettings(reranker_enabled=True, reranker_provider="flashrank")
    with pytest.raises(NotImplementedError):
        RerankerFactory.create_reranker(cfg_flash)

    # 6. Invalid provider
    with pytest.raises(ValidationError):
        # type ignore for testing invalid provider input
        RetrievalSettings(reranker_enabled=True, reranker_provider="invalid")  # type: ignore[arg-type]
