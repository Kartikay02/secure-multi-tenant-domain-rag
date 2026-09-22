"""Vector store interfaces and protocol definitions."""

import uuid
from typing import Any, Protocol, runtime_checkable

from app.rag.embeddings.vector_interfaces import VectorRecord
from app.rag.vector.domain import RetrievalResult


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Abstract interface for vector database storage and similarity search backends."""

    async def upsert_vectors(self, records: list[VectorRecord]) -> int:
        """Insert or update a list of vector records.

        Args:
            records: List of VectorRecord items containing id (chunk_id), vector, document_id, metadata.

        Returns:
            Number of records successfully persisted.
        """
        ...

    async def similarity_search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Perform nearest-neighbor vector similarity search with metadata filtering.

        Args:
            query_vector: Query embedding dense vector.
            top_k: Maximum number of candidates to return.
            score_threshold: Minimum normalized similarity score [0.0, 1.0].
            filter_metadata: Dictionary of metadata constraints (e.g. document_id, tags).

        Returns:
            List of RetrievalResult objects sorted descending by score.
        """
        ...

    async def delete_vectors(self, chunk_ids: list[uuid.UUID]) -> int:
        """Delete vectors by chunk IDs."""
        ...

    async def delete_by_document(self, document_id: uuid.UUID) -> int:
        """Delete all vectors belonging to a document."""
        ...

    async def get_vector(self, chunk_id: uuid.UUID) -> VectorRecord | None:
        """Fetch a stored vector record by chunk ID."""
        ...

    async def count(self) -> int:
        """Return total count of indexed vectors."""
        ...
