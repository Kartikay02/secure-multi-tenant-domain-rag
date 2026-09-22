"""No-op pass-through reranker when reranking is disabled."""

from collections.abc import Sequence

from app.rag.reranking.interfaces import RerankerProtocol
from app.rag.vector.domain import RetrievalResult


class NoOpReranker(RerankerProtocol):
    """Pass-through reranker that returns candidate slices without re-scoring."""

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Return candidates sliced to top_k without re-scoring.

        Args:
            query: The search query text (ignored in no-op).
            candidates: Retrieved candidate chunks.
            top_k: Maximum candidates to return.

        Returns:
            List of candidates up to top_k.
        """
        return list(candidates[:top_k])
