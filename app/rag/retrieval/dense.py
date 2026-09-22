import hashlib
from typing import Any

from app.core.logging import get_logger
from app.rag.embeddings.interfaces import EmbeddingProtocol
from app.rag.retrieval.interfaces import RetrieverProtocol
from app.rag.vector.domain import RetrievalResult
from app.rag.vector.interfaces import VectorStoreProtocol

logger = get_logger("app.rag.retrieval.dense")


class DenseRetriever(RetrieverProtocol):
    """Retrieves semantically similar chunks using dense vector embeddings."""

    def __init__(
        self,
        embedding_provider: EmbeddingProtocol,
        vector_store: VectorStoreProtocol,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve candidate chunks by dense vector similarity."""
        if not query or not query.strip():
            return []

        clean_query = query.strip()
        query_fp = hashlib.sha256(clean_query.encode("utf-8")).hexdigest()[:8]
        logger.debug(
            f"DenseRetriever generating vector for query_fp={query_fp} (len={len(clean_query)})",
            extra={"query_hash": query_fp, "query_length": len(clean_query)},
        )
        query_vector = await self.embedding_provider.embed_text(clean_query)

        raw_results = await self.vector_store.similarity_search(
            query_vector=query_vector,
            top_k=top_k,
            score_threshold=score_threshold,
            filter_metadata=filter_metadata,
        )

        # Ensure dense_score is stamped on every result
        dense_results = [
            RetrievalResult(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                text=r.text,
                score=r.score,
                dense_score=r.score,
                sparse_score=r.sparse_score,
                metadata=r.metadata,
            )
            for r in raw_results
        ]
        logger.debug(f"DenseRetriever returned {len(dense_results)} candidates")
        return dense_results
