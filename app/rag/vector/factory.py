"""Vector store factory instantiating configured storage backends."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.rag.vector.in_memory import InMemoryVectorStore
from app.rag.vector.interfaces import VectorStoreProtocol
from app.rag.vector.pgvector import PGVectorStore

# Singleton in-memory vector store shared across session instances in mock mode
_in_memory_store: InMemoryVectorStore | None = None


class VectorStoreFactory:
    """Factory selecting and configuring VectorStoreProtocol instances."""

    @classmethod
    def create(
        cls,
        settings: Settings,
        session: AsyncSession | None = None,
    ) -> VectorStoreProtocol:
        """Provide appropriate vector store backend based on configuration (SEC-43)."""
        raw_provider = (
            getattr(settings, "vector_store_provider", None)
            or getattr(getattr(settings, "vector_store", None), "provider", None)
            or "pgvector"
        )
        provider = str(raw_provider).lower().strip()

        if provider in ("mock", "memory", "in_memory"):
            global _in_memory_store
            if _in_memory_store is None:
                _in_memory_store = InMemoryVectorStore(dimension=settings.embeddings.dimension)
            return _in_memory_store

        elif provider == "pgvector":
            if settings.database.url.startswith("sqlite"):
                raise ConfigurationError(
                    "VECTOR_STORE_PROVIDER='pgvector' is incompatible with SQLite databases. "
                    "Set VECTOR_STORE_PROVIDER='mock' for local development with SQLite."
                )
            return PGVectorStore(
                session=session,
                model_name=settings.embeddings.model,
                dimension=settings.embeddings.dimension,
            )

        else:
            raise ConfigurationError(
                f"Unsupported VECTOR_STORE_PROVIDER '{provider}'. Supported: 'pgvector', 'mock'."
            )

    @classmethod
    def reset_mock(cls) -> None:
        """Reset mock in-memory vector store for testing."""
        global _in_memory_store
        _in_memory_store = None
