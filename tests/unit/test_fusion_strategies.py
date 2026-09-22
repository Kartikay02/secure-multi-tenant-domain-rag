import asyncio
import time
import uuid
from unittest.mock import AsyncMock

import pytest

from app.rag.retrieval.fusion import (
    LinearCombinationFusion,
    ReciprocalRankFusion,
    RelativeScoreFusion,
)
from app.rag.retrieval.hybrid import HybridRetriever
from app.rag.vector.domain import RetrievalResult


def _make_result(chunk_id: uuid.UUID, score: float, text: str = "text") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=uuid.uuid4(),
        text=text,
        score=score,
        metadata={"orig": True},
    )


def test_rrf_cross_modal_agreement() -> None:
    """Verify item appearing in both dense and sparse branches gets highest fused rank."""
    c1 = uuid.uuid4()
    c2 = uuid.uuid4()
    c3 = uuid.uuid4()

    # c1 is rank 1 in dense, rank 1 in sparse
    # c2 is rank 2 in dense, absent in sparse
    # c3 is rank 2 in sparse, absent in dense
    dense_results = [_make_result(c1, 0.95), _make_result(c2, 0.85)]
    sparse_results = [_make_result(c1, 0.90), _make_result(c3, 0.80)]

    rrf = ReciprocalRankFusion(k=60)
    fused = rrf.fuse(dense_results, sparse_results, top_k=5)

    assert len(fused) == 3
    # c1 must be rank 1 due to agreement in both modalities
    assert fused[0].chunk_id == c1
    # Check score attribution
    assert fused[0].dense_score == 0.95
    assert fused[0].sparse_score == 0.90
    assert fused[0].score > fused[1].score
    assert fused[0].metadata["fusion_method"] == "rrf"


def test_rrf_deduplication() -> None:
    """Verify duplicate chunks across dense and sparse are deduplicated."""
    shared_id = uuid.uuid4()
    dense_results = [_make_result(shared_id, 0.9)]
    sparse_results = [_make_result(shared_id, 0.8)]

    rrf = ReciprocalRankFusion(k=60)
    fused = rrf.fuse(dense_results, sparse_results, top_k=5)

    assert len(fused) == 1
    assert fused[0].chunk_id == shared_id
    assert fused[0].dense_score == 0.9
    assert fused[0].sparse_score == 0.8


def test_rrf_top_k_limiting() -> None:
    """Verify top_k limit is respected."""
    dense = [_make_result(uuid.uuid4(), 0.9 - i * 0.05) for i in range(10)]
    sparse = [_make_result(uuid.uuid4(), 0.9 - i * 0.05) for i in range(10)]

    rrf = ReciprocalRankFusion(k=60)
    fused = rrf.fuse(dense, sparse, top_k=4)
    assert len(fused) == 4


def test_linear_combination_fusion() -> None:
    """Verify weighted linear combination blends dense and sparse scores."""
    c1 = uuid.uuid4()
    c2 = uuid.uuid4()

    # c1 in both: 0.7 * 1.0 + 0.3 * 0.5 = 0.7 + 0.15 = 0.85
    # c2 in dense only: 0.7 * 0.8 = 0.56
    dense = [_make_result(c1, 1.0), _make_result(c2, 0.8)]
    sparse = [_make_result(c1, 0.5)]

    linear = LinearCombinationFusion(dense_weight=0.7, sparse_weight=0.3)
    fused = linear.fuse(dense, sparse, top_k=5)

    assert len(fused) == 2
    assert fused[0].chunk_id == c1
    assert fused[0].score == pytest.approx(0.85, rel=1e-3)
    assert fused[0].dense_score == 1.0
    assert fused[0].sparse_score == 0.5

    assert fused[1].chunk_id == c2
    assert fused[1].score == pytest.approx(0.56, rel=1e-3)
    assert fused[1].dense_score == 0.8
    assert fused[1].sparse_score is None


def test_relative_score_fusion() -> None:
    """Verify relative score fusion scales each list by its max score before blending."""
    c1 = uuid.uuid4()
    c2 = uuid.uuid4()

    # dense scores: c1=0.5 (max is 0.5 -> scaled to 1.0)
    # sparse scores: c2=10.0 (max is 10.0 -> scaled to 1.0)
    dense = [_make_result(c1, 0.5)]
    sparse = [_make_result(c2, 10.0)]

    # weights: 0.5 each
    rel = RelativeScoreFusion(dense_weight=0.5, sparse_weight=0.5)
    fused = rel.fuse(dense, sparse, top_k=5)

    assert len(fused) == 2
    # Both should have normalized score of 0.5
    assert fused[0].score == pytest.approx(0.5, rel=1e-3)
    assert fused[1].score == pytest.approx(0.5, rel=1e-3)
    assert fused[0].metadata["fusion_method"] == "relative_score"


class TestHybridRetrieverConcurrencyAndResilience:
    """Test suite for HybridRetriever safe asyncio.gather concurrency, failure degradation, and bounds."""

    @pytest.mark.asyncio
    async def test_concurrent_execution_timing(self) -> None:
        """Dense and lexical execute concurrently with asyncio.gather, completing in parallel time."""
        c1 = uuid.uuid4()
        c2 = uuid.uuid4()

        class SlowDenseRetriever:
            async def retrieve(self, *args, **kwargs):
                await asyncio.sleep(0.06)
                return [_make_result(c1, 0.9)]

        class SlowLexicalRetriever:
            async def retrieve(self, *args, **kwargs):
                await asyncio.sleep(0.06)
                return [_make_result(c2, 0.8)]

        hybrid = HybridRetriever(
            dense_retriever=SlowDenseRetriever(),
            lexical_retriever=SlowLexicalRetriever(),
        )

        t0 = time.perf_counter()
        results = await hybrid.retrieve("Concurrency query")
        elapsed = time.perf_counter() - t0

        assert len(results) == 2
        # If sequential, elapsed would be >= 0.12s; concurrent must be < 0.11s
        assert elapsed < 0.11

    @pytest.mark.asyncio
    async def test_dense_failure_gracefully_falls_back_to_lexical(self) -> None:
        """If dense retrieval raises an exception, hybrid succeeds with lexical results."""
        from app.rag.retrieval.hybrid import HybridRetriever

        c1 = uuid.uuid4()

        class FailingDenseRetriever:
            async def retrieve(self, *args, **kwargs):
                raise ConnectionError("Vector database connection lost")

        class HealthyLexicalRetriever:
            async def retrieve(self, *args, **kwargs):
                return [_make_result(c1, 0.85)]

        hybrid = HybridRetriever(
            dense_retriever=FailingDenseRetriever(),
            lexical_retriever=HealthyLexicalRetriever(),
        )

        results = await hybrid.retrieve("Resilience test")
        assert len(results) == 1
        assert results[0].chunk_id == c1

    @pytest.mark.asyncio
    async def test_lexical_failure_gracefully_falls_back_to_dense(self) -> None:
        """If lexical retrieval raises an exception, hybrid succeeds with dense results."""
        from app.rag.retrieval.hybrid import HybridRetriever

        c1 = uuid.uuid4()

        class HealthyDenseRetriever:
            async def retrieve(self, *args, **kwargs):
                return [_make_result(c1, 0.92)]

        class FailingLexicalRetriever:
            async def retrieve(self, *args, **kwargs):
                raise TimeoutError("Full-text search timed out")

        hybrid = HybridRetriever(
            dense_retriever=HealthyDenseRetriever(),
            lexical_retriever=FailingLexicalRetriever(),
        )

        results = await hybrid.retrieve("Resilience test")
        assert len(results) == 1
        assert results[0].chunk_id == c1

    @pytest.mark.asyncio
    async def test_both_branches_failing_raises_retrieval_error(self) -> None:
        """If both dense and lexical fail, HybridRetriever raises RetrievalError."""
        from app.core.exceptions import RetrievalError
        from app.rag.retrieval.hybrid import HybridRetriever

        class FailingDense:
            async def retrieve(self, *args, **kwargs):
                raise ConnectionError("Dense down")

        class FailingLexical:
            async def retrieve(self, *args, **kwargs):
                raise ConnectionError("Lexical down")

        hybrid = HybridRetriever(
            dense_retriever=FailingDense(),
            lexical_retriever=FailingLexical(),
        )

        with pytest.raises(RetrievalError) as exc_info:
            await hybrid.retrieve("Complete outage test")

        assert "Both dense and lexical retrieval branches failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_tenant_filter_preservation(self) -> None:
        """Tenant filter metadata is faithfully propagated to both branches."""
        c1 = uuid.uuid4()
        mock_dense = AsyncMock()
        mock_dense.retrieve = AsyncMock(return_value=[_make_result(c1, 0.9)])
        mock_lexical = AsyncMock()
        mock_lexical.retrieve = AsyncMock(return_value=[_make_result(c1, 0.8)])

        hybrid = HybridRetriever(
            dense_retriever=mock_dense,
            lexical_retriever=mock_lexical,
        )

        filters = {"tenant_id": "tenant-finance-42"}
        await hybrid.retrieve("Query with filter", filter_metadata=filters)

        mock_dense.retrieve.assert_called_once()
        assert mock_dense.retrieve.call_args[1]["filter_metadata"] == filters
        mock_lexical.retrieve.assert_called_once()
        assert mock_lexical.retrieve.call_args[1]["filter_metadata"] == filters
