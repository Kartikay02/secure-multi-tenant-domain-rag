"""Vector storage abstraction and DTOs to decouple embedding logic from storage backends."""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class VectorRecord:
    """Represents a vector embedding with associated document chunk metadata."""

    id: str
    vector: list[float]
    document_id: str = ""
    version_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Abstract interface for vector database storage and indexing backends.

    Allows EmbeddingService and Retrieval to interact with vector storage
    without coupling to pgvector, Qdrant, Milvus, or Chroma.
    """

    async def upsert(self, records: list[VectorRecord]) -> int:
        """Insert or update a list of vector records.

        Args:
            records: List of VectorRecord instances to persist.

        Returns:
            Count of successfully stored records.
        """
        ...

    async def get(self, record_id: str) -> VectorRecord | None:
        """Retrieve a vector record by its identifier."""
        ...

    async def delete(self, record_ids: list[str]) -> int:
        """Delete vector records by their identifiers.

        Args:
            record_ids: Identifiers of records to remove.

        Returns:
            Count of deleted records.
        """
        ...

    async def count(self) -> int:
        """Count total vectors indexed in the store."""
        ...
