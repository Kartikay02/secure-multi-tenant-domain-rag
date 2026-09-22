"""Domain service for dense vector retrieval."""

from typing import Any

from app.core.logging import get_logger
from app.rag.embeddings.interfaces import EmbeddingProtocol
from app.rag.vector.domain import RetrievalResult
from app.rag.vector.interfaces import VectorStoreProtocol

logger = get_logger("app.services.retrieval")


class VectorRetrievalService:
    """Orchestrates query embedding generation and vector similarity search."""

    def __init__(
        self,
        embedding_provider: EmbeddingProtocol,
        vector_store: VectorStoreProtocol,
        default_top_k: int = 5,
        default_score_threshold: float | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.default_top_k = default_top_k
        self.default_score_threshold = default_score_threshold

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Perform dense vector retrieval for a natural language query.

        Args:
            query: Natural language query string.
            top_k: Optional maximum number of candidate chunks to return.
            score_threshold: Optional minimum normalized similarity threshold [0.0, 1.0].
            filter_metadata: Optional dictionary of metadata constraints.

        Returns:
            List of RetrievalResult objects sorted descending by similarity score.
        """
        if not query or not query.strip():
            return []

        resolved_top_k = top_k if top_k is not None else self.default_top_k
        resolved_threshold = (
            score_threshold if score_threshold is not None else self.default_score_threshold
        )

        logger.info(
            f"Executing vector retrieval for query (top_k={resolved_top_k}, threshold={resolved_threshold})",
            extra={"query_length": len(query), "top_k": resolved_top_k},
        )

        # 1. Embed query text
        query_vector = await self.embedding_provider.embed_text(query.strip())

        # 2. Similarity search in vector store
        results = await self.vector_store.similarity_search(
            query_vector=query_vector,
            top_k=resolved_top_k,
            score_threshold=resolved_threshold,
            filter_metadata=filter_metadata,
        )

        logger.info(
            f"Vector search yielded {len(results)} candidate chunks",
            extra={"result_count": len(results)},
        )
        return results
