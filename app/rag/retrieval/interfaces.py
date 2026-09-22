"""Retrieval abstractions and protocol contracts."""

from typing import Any, Protocol, runtime_checkable

from app.rag.vector.domain import RetrievalResult


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Protocol for single-modality retrievers (dense, lexical, etc.)."""

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Execute candidate chunk retrieval for a query.

        Args:
            query: Query string.
            top_k: Max candidate chunks to return.
            score_threshold: Optional minimum similarity threshold [0.0, 1.0].
            filter_metadata: Optional dictionary of metadata constraints.

        Returns:
            List of RetrievalResult objects ordered by relevance score.
        """
        ...


@runtime_checkable
class FusionStrategyProtocol(Protocol):
    """Protocol for algorithms combining multi-modal retrieval lists into a unified ranking."""

    def fuse(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Combine dense and sparse candidate lists into a single deduplicated ranking.

        Args:
            dense_results: Ranked candidates from dense vector search.
            sparse_results: Ranked candidates from sparse lexical search.
            top_k: Maximum number of final unified chunks to produce.

        Returns:
            List of deduplicated RetrievalResult objects ordered by fused score.
        """
        ...
