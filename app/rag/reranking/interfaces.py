"""Protocol contracts for cross-encoder reranking."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from app.rag.vector.domain import RetrievalResult


@runtime_checkable
class RerankerProtocol(Protocol):
    """Protocol for cross-encoder rerankers scoring (query, document) candidate pairs."""

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Rerank retrieval candidate chunks and return top_k results.

        Args:
            query: The search query text.
            candidates: Candidate chunks retrieved from prior search stages.
            top_k: Maximum number of reranked results to return.

        Returns:
            List of RetrievalResult objects sorted descending by rerank score.
        """
        ...
