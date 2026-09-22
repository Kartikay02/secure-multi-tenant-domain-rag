"""Pytest fixtures and test environment configuration."""

import io
import uuid
from collections.abc import AsyncIterator, Sequence
from typing import Any

import docx
import pypdf
import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.api.deps import get_app_settings
from app.core.config import (
    ApplicationSettings,
    DatabaseSettings,
    ObservabilitySettings,
    SecuritySettings,
    Settings,
    VectorStoreSettings,
)
from app.core.exceptions import (
    DatabaseError,
    EmbeddingError,
    EmbeddingTimeoutError,
    GenerationError,
    GenerationTimeoutError,
    RerankError,
    RerankTimeoutError,
)
from app.main import create_app
from app.models import Base
from app.rag.embeddings.interfaces import EmbeddingProtocol, EmbeddingResult
from app.rag.embeddings.vector_interfaces import VectorRecord
from app.rag.generation.domain import GeneratedResponse, GenerationParameters
from app.rag.generation.interfaces import LLMProviderProtocol
from app.rag.reranking.interfaces import RerankerProtocol
from app.rag.vector.domain import RetrievalResult
from app.rag.vector.interfaces import VectorStoreProtocol
from app.repositories import (
    SQLAlchemyDocumentChunkRepository,
    SQLAlchemyDocumentRepository,
    SQLAlchemyDocumentVersionRepository,
    SQLAlchemyIngestionJobRepository,
)


@pytest.fixture
def test_settings() -> Settings:
    """Fixture providing isolated test settings with sections."""
    return Settings(
        app=ApplicationSettings(
            name="Test Domain RAG",
            env="test",
            debug=True,
        ),
        database=DatabaseSettings(
            url="sqlite+aiosqlite:///:memory:",
        ),
        vector_store=VectorStoreSettings(
            provider="mock",
        ),
        security=SecuritySettings(
            api_key="test-api-key",
        ),
        observability=ObservabilitySettings(
            log_level="DEBUG",
            log_format="console",
        ),
    )


@pytest.fixture
async def async_client(
    test_settings: Settings,
    db_engine: AsyncEngine,
    tmp_path_factory: pytest.TempPathFactory,
) -> AsyncIterator[AsyncClient]:
    """Fixture providing an async test client bound to the FastAPI test app."""
    from app.api.deps import get_storage_service
    from app.core.database import get_db, reset_current_session, set_current_session
    from app.rag.storage.local import LocalStorageService

    upload_dir = tmp_path_factory.mktemp("uploads")
    storage = LocalStorageService(base_dir=upload_dir)

    session_factory = async_sessionmaker(
        bind=db_engine,
        class_=AsyncSession,
        autoflush=False,
        expire_on_commit=False,
    )

    async def override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            token = set_current_session(session)
            try:
                yield session
                await session.commit()
            finally:
                reset_current_session(token)

    app = create_app(settings=test_settings)
    app.dependency_overrides[get_app_settings] = lambda: test_settings
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_storage_service] = lambda: storage

    transport = ASGITransport(app=app)
    headers = {"X-API-Key": test_settings.security.api_key}
    async with AsyncClient(
        transport=transport, base_url="http://testserver", headers=headers
    ) as client:
        yield client


@pytest.fixture
async def db_engine() -> AsyncIterator[AsyncEngine]:
    """Provide an in-memory SQLite async engine with all tables created."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """Provide a transactional async session bound to the test in-memory database."""
    session_factory = async_sessionmaker(
        bind=db_engine,
        class_=AsyncSession,
        autoflush=False,
        expire_on_commit=False,
    )
    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest.fixture
def document_repo(db_session: AsyncSession) -> SQLAlchemyDocumentRepository:
    """Fixture providing Document repository bound to the test session."""
    return SQLAlchemyDocumentRepository(session=db_session)


@pytest.fixture
def version_repo(db_session: AsyncSession) -> SQLAlchemyDocumentVersionRepository:
    """Fixture providing DocumentVersion repository bound to the test session."""
    return SQLAlchemyDocumentVersionRepository(session=db_session)


@pytest.fixture
def chunk_repo(db_session: AsyncSession) -> SQLAlchemyDocumentChunkRepository:
    """Fixture providing DocumentChunk repository bound to the test session."""
    return SQLAlchemyDocumentChunkRepository(session=db_session)


@pytest.fixture
def job_repo(db_session: AsyncSession) -> SQLAlchemyIngestionJobRepository:
    """Fixture providing IngestionJob repository bound to the test session."""
    return SQLAlchemyIngestionJobRepository(session=db_session)


class FailingEmbeddingProvider(EmbeddingProtocol):
    """Embedding provider that simulates upstream failures or timeouts."""

    def __init__(self, mode: str = "error") -> None:
        self.mode = mode

    @property
    def dimension(self) -> int:
        return 384

    @property
    def model_name(self) -> str:
        return "failing-embed-model"

    async def embed_text(self, text: str) -> list[float]:
        if self.mode == "timeout":
            raise EmbeddingTimeoutError(provider="failing-embed", timeout_seconds=5.0)
        raise EmbeddingError("Simulated upstream embedding provider failure (502 Bad Gateway).")

    async def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[EmbeddingResult]:
        if self.mode == "timeout":
            raise EmbeddingTimeoutError(provider="failing-embed", timeout_seconds=5.0)
        raise EmbeddingError("Simulated upstream embedding provider failure (502 Bad Gateway).")

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.mode == "timeout":
            raise EmbeddingTimeoutError(provider="failing-embed", timeout_seconds=5.0)
        raise EmbeddingError("Simulated upstream embedding provider failure (502 Bad Gateway).")


class FailingVectorStore(VectorStoreProtocol):
    """Vector store that simulates database failure during operations."""

    async def upsert(self, records: list[VectorRecord]) -> int:
        raise DatabaseError("Simulated vector storage failure during upsert.")

    async def upsert_vectors(self, records: list[VectorRecord]) -> int:
        raise DatabaseError("Simulated vector storage failure during upsert_vectors.")

    async def get(self, record_id: str) -> VectorRecord | None:
        raise DatabaseError("Simulated vector storage failure during get.")

    async def delete(self, record_ids: list[str]) -> int:
        raise DatabaseError("Simulated vector storage failure during delete.")

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        score_threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[VectorRecord]:
        raise DatabaseError("Simulated vector database connection failure during search.")

    async def similarity_search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        score_threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        raise DatabaseError(
            "Simulated vector database connection failure during similarity search."
        )

    async def delete_vectors(self, chunk_ids: list[uuid.UUID]) -> int:
        raise DatabaseError("Simulated vector database failure during delete_vectors.")

    async def delete_by_document(self, document_id: uuid.UUID) -> int:
        raise DatabaseError("Simulated vector database failure during delete_by_document.")

    async def get_vector(self, chunk_id: uuid.UUID) -> VectorRecord | None:
        raise DatabaseError("Simulated vector database failure during get_vector.")

    async def count(self) -> int:
        raise DatabaseError("Simulated vector database failure during count.")


class FailingReranker(RerankerProtocol):
    """Cross-encoder reranker simulating timeout or unhandled upstream failure."""

    def __init__(self, mode: str = "error") -> None:
        self.mode = mode

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if self.mode == "timeout":
            raise RerankTimeoutError(provider="mock-reranker", timeout_seconds=2.0)
        raise RerankError("Simulated cross-encoder reranking upstream failure.")


class FailingLLMProvider(LLMProviderProtocol):
    """LLM provider simulating timeout, rate limit, or generation failure."""

    def __init__(self, mode: str = "timeout") -> None:
        self.mode = mode

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        parameters: GenerationParameters | None = None,
        structured_schema: type[BaseModel] | None = None,
    ) -> GeneratedResponse:
        if self.mode == "timeout":
            raise GenerationTimeoutError(provider="mock-llm", timeout_seconds=10.0)
        raise GenerationError("Simulated LLM generation API failure.")

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        parameters: GenerationParameters | None = None,
    ) -> AsyncIterator[str]:
        if self.mode == "timeout":
            raise GenerationTimeoutError(provider="mock-llm", timeout_seconds=10.0)
        raise GenerationError("Simulated streaming LLM failure.")
        if False:
            yield ""


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Fixture providing valid minimal PDF binary bytes."""
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.fixture
def sample_docx_bytes() -> bytes:
    """Fixture providing valid minimal DOCX binary bytes."""
    doc = docx.Document()
    doc.add_paragraph("Reliable consensus protocols in distributed systems.")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


@pytest.fixture
def corrupted_pdf_bytes() -> bytes:
    """Fixture providing corrupted PDF bytes."""
    return b"%PDF-invalid-corrupted-stream"


@pytest.fixture
def corrupted_docx_bytes() -> bytes:
    """Fixture providing corrupted DOCX bytes."""
    return b"PK\x03\x04invalid-corrupted-archive"
