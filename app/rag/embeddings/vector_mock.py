"""In-memory vector store implementation of VectorStoreProtocol for testing."""

from app.core.logging import get_logger
from app.rag.embeddings.vector_interfaces import VectorRecord, VectorStoreProtocol

logger = get_logger("app.rag.embeddings.vector_mock")


class InMemoryVectorStore(VectorStoreProtocol):
    """Simple in-memory vector store for testing and pipeline decoupling."""

    def __init__(self) -> None:
        self._store: dict[str, VectorRecord] = {}

    async def upsert(self, records: list[VectorRecord]) -> int:
        """Store or update vector records."""
        count = 0
        for r in records:
            self._store[r.id] = r
            count += 1
        logger.debug(
            f"Upserted {count} records into InMemoryVectorStore. Total: {len(self._store)}"
        )
        return count

    async def get(self, record_id: str) -> VectorRecord | None:
        """Retrieve a record by ID."""
        return self._store.get(record_id)

    async def delete(self, record_ids: list[str]) -> int:
        """Remove records by IDs."""
        deleted = 0
        for rid in record_ids:
            if rid in self._store:
                del self._store[rid]
                deleted += 1
        return deleted

    async def count(self) -> int:
        """Return total number of records."""
        return len(self._store)

    def clear(self) -> None:
        """Clear all stored vectors."""
        self._store.clear()
