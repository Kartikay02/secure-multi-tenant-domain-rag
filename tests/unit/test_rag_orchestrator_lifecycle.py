"""Unit tests for RAG orchestrator singleton lifecycle and session isolation (Issue 2)."""

import asyncio
from unittest.mock import MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    get_rag_orchestrator,
    init_rag_components,
    shutdown_rag_components,
)
from app.core.config import Settings
from app.core.database import (
    get_current_session,
    reset_current_session,
    set_current_session,
)
from app.rag.retrieval.lexical import PGLexicalRetriever
from app.rag.vector.pgvector import PGVectorStore


@pytest.fixture(autouse=True)
def clean_rag_lifecycle():
    """Ensure RAG components are cleanly torn down before and after each test."""
    shutdown_rag_components()
    yield
    shutdown_rag_components()


def test_rag_orchestrator_singleton_lifecycle() -> None:
    """Verify RAGOrchestratorService is instantiated as a singleton and reused."""
    settings = Settings()
    # Force mock vector store so test runs without live postgres
    settings.vector_store.provider = "mock"

    orch1 = init_rag_components(settings)
    orch2 = get_rag_orchestrator(settings=settings)
    orch3 = get_rag_orchestrator(settings=settings)

    assert orch1 is orch2
    assert orch2 is orch3

    shutdown_rag_components()
    orch4 = get_rag_orchestrator(settings=settings)
    assert orch4 is not orch1


@pytest.mark.asyncio
async def test_session_isolation_via_contextvars() -> None:
    """Verify concurrent async tasks maintain isolated database sessions via contextvars."""
    session_1 = MagicMock(spec=AsyncSession)
    session_2 = MagicMock(spec=AsyncSession)

    task_1_sessions: list[AsyncSession | None] = []
    task_2_sessions: list[AsyncSession | None] = []

    async def worker_1() -> None:
        token = set_current_session(session_1)
        try:
            await asyncio.sleep(0.01)
            task_1_sessions.append(get_current_session())
            await asyncio.sleep(0.01)
            task_1_sessions.append(get_current_session())
        finally:
            reset_current_session(token)

    async def worker_2() -> None:
        token = set_current_session(session_2)
        try:
            await asyncio.sleep(0.01)
            task_2_sessions.append(get_current_session())
            await asyncio.sleep(0.01)
            task_2_sessions.append(get_current_session())
        finally:
            reset_current_session(token)

    assert get_current_session() is None
    await asyncio.gather(worker_1(), worker_2())
    assert get_current_session() is None

    assert task_1_sessions == [session_1, session_1]
    assert task_2_sessions == [session_2, session_2]
    assert task_1_sessions[0] is not task_2_sessions[0]


def test_retriever_dynamic_session_resolution() -> None:
    """Verify PGLexicalRetriever and PGVectorStore dynamically resolve task session."""
    mock_session = MagicMock(spec=AsyncSession)
    token = set_current_session(mock_session)

    try:
        retriever = PGLexicalRetriever(session=None)
        assert retriever.session is mock_session

        store = PGVectorStore(session=None)
        assert store.session is mock_session
    finally:
        reset_current_session(token)

    # When contextvar is unset and no explicit session provided, accessing session raises DatabaseError
    from app.core.exceptions import DatabaseError

    retriever_unbound = PGLexicalRetriever(session=None)
    with pytest.raises(DatabaseError, match="No active database session available"):
        _ = retriever_unbound.session


def test_embedding_provider_singleton_reused_across_requests() -> None:
    """Verify get_embedding_provider returns a singleton provider and reuses it (Issue 6)."""
    from app.api.deps import get_embedding_provider

    settings = Settings()
    prov1 = get_embedding_provider(settings)
    prov2 = get_embedding_provider(settings)

    assert prov1 is prov2

    shutdown_rag_components()
    prov3 = get_embedding_provider(settings)
    assert prov3 is not prov1


def test_embedding_service_reuses_provider_with_isolated_db_sessions() -> None:
    """Verify get_embedding_service reuses the provider while isolating request sessions."""
    from app.api.deps import get_embedding_provider, get_embedding_service
    from app.core.config import DatabaseSettings, VectorStoreSettings

    settings = Settings(
        database=DatabaseSettings(url="postgresql+asyncpg://postgres:postgres@localhost:5432/test_db"),
        vector_store=VectorStoreSettings(provider="pgvector"),
    )
    session_a = MagicMock(spec=AsyncSession)
    session_b = MagicMock(spec=AsyncSession)

    doc_repo = MagicMock()
    ver_repo = MagicMock()
    chunk_repo = MagicMock()
    job_repo = MagicMock()

    provider = get_embedding_provider(settings)

    svc_a = get_embedding_service(
        doc_repo=doc_repo,
        ver_repo=ver_repo,
        chunk_repo=chunk_repo,
        job_repo=job_repo,
        session=session_a,
        settings=settings,
        embedding_provider=provider,
    )

    svc_b = get_embedding_service(
        doc_repo=doc_repo,
        ver_repo=ver_repo,
        chunk_repo=chunk_repo,
        job_repo=job_repo,
        session=session_b,
        settings=settings,
        embedding_provider=provider,
    )

    # Embedding provider must be the identical shared instance
    assert svc_a.embedding_provider is svc_b.embedding_provider
    # But underlying vector stores have isolated DB sessions
    assert svc_a.vector_store.session is session_a  # type: ignore[union-attr]
    assert svc_b.vector_store.session is session_b  # type: ignore[union-attr]
    assert svc_a.vector_store.session is not svc_b.vector_store.session  # type: ignore[union-attr]
