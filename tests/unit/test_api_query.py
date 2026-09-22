"""Unit tests for query endpoints: POST /query, POST /query/stream, GET /health, GET /ready."""

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import AsyncClient

from app.api.deps import get_rag_orchestrator
from app.rag.context.domain import ContextDocument
from app.rag.orchestration.domain import PipelineStageLatency, RAGResponse


@pytest.fixture
def mock_rag_response() -> RAGResponse:
    doc_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    doc = ContextDocument(
        citation_id=1,
        citation_label="[1]",
        chunk_id=chunk_id,
        document_id=doc_id,
        source_id="guide.md",
        title="Deployment Guide",
        page_number=1,
        score=0.91,
        content="Kubernetes deployments require liveness and readiness probes.",
        token_count=10,
    )
    return RAGResponse(
        query="What probes does Kubernetes need?",
        raw_query="What probes does Kubernetes need?",
        answer="Kubernetes deployments require liveness and readiness probes [1].",
        grounded=True,
        confidence_score=0.95,
        insufficient_context=False,
        fallback_applied=False,
        citations=[1],
        referenced_documents=[doc],
        citation_manifest='References:\n[1] "Deployment Guide" (Source: guide.md)',
        unsupported_claims=[],
        citation_errors=[],
        retrieved_chunks_count=1,
        context_chunks_count=1,
        context_tokens=10,
        stage_latencies=PipelineStageLatency(
            preprocessing_ms=1.2,
            retrieval_ms=15.0,
            reranking_ms=4.5,
            context_assembly_ms=3.0,
            generation_ms=25.0,
            validation_ms=2.0,
            total_ms=50.7,
        ),
        request_id="trace-req-001",
        model="mock-gpt-4o",
        metadata={"model": "mock-gpt-4o"},
    )


@pytest.mark.asyncio
async def test_query_happy_path(
    async_client: AsyncClient,
    mock_rag_response: RAGResponse,
) -> None:
    """Verify POST /api/v1/query executes RAG query and returns structured QueryResponse."""
    mock_orchestrator = MagicMock()
    mock_orchestrator.execute = AsyncMock(return_value=mock_rag_response)

    # Override orchestrator dependency on the app
    app = async_client._transport.app  # type: ignore[attr-defined]
    app.dependency_overrides[get_rag_orchestrator] = lambda: mock_orchestrator

    try:
        response = await async_client.post(
            "/api/v1/query",
            json={
                "query": "What probes does Kubernetes need?",
                "top_k": 5,
                "temperature": 0.0,
            },
        )
        assert response.status_code == 200
        data = response.json()

        assert data["answer"] == "Kubernetes deployments require liveness and readiness probes [1]."
        assert data["grounded"] is True
        assert data["confidence_score"] == 0.95
        assert data["citations"] == [1]
        assert len(data["referenced_documents"]) == 1
        assert data["referenced_documents"][0]["citation_id"] == 1
        assert data["referenced_documents"][0]["title"] == "Deployment Guide"
        assert data["request_id"] == "trace-req-001"
        assert "retrieval_metadata" in data
    finally:
        app.dependency_overrides.pop(get_rag_orchestrator, None)


@pytest.mark.asyncio
async def test_query_validation_failure(async_client: AsyncClient) -> None:
    """Verify POST /api/v1/query with empty query returns HTTP 422 with RFC 7807 ErrorResponse."""
    response = await async_client.post(
        "/api/v1/query",
        json={"query": ""},
    )
    assert response.status_code == 422
    data = response.json()
    assert data["error_code"] == "VALIDATION_ERROR"
    assert len(data["errors"]) >= 1


@pytest.mark.asyncio
async def test_query_timeout_handling(async_client: AsyncClient) -> None:
    """Verify POST /api/v1/query returns HTTP 504 on provider/pipeline timeout."""

    async def slow_execute(**kwargs: Any) -> RAGResponse:
        await asyncio.sleep(5.0)
        raise TimeoutError("Simulated timeout")

    mock_orchestrator = MagicMock()
    mock_orchestrator.execute = slow_execute

    # Set low timeout settings override
    from app.core.config import Settings

    app = async_client._transport.app  # type: ignore[attr-defined]
    orig_orchestrator = app.dependency_overrides.get(get_rag_orchestrator)
    app.dependency_overrides[get_rag_orchestrator] = lambda: mock_orchestrator

    from app.api.deps import get_app_settings

    orig_settings = app.dependency_overrides.get(get_app_settings)

    # Settings with 0.05s timeout
    fast_timeout_settings = Settings()
    fast_timeout_settings.llm.timeout_seconds = 0.05
    app.dependency_overrides[get_app_settings] = lambda: fast_timeout_settings

    try:
        response = await async_client.post(
            "/api/v1/query",
            json={"query": "Test slow query"},
        )
        assert response.status_code == 504
        data = response.json()
        assert data["error_code"] == "QUERY_TIMEOUT"
        assert "timed out" in data["detail"]
    finally:
        if orig_orchestrator:
            app.dependency_overrides[get_rag_orchestrator] = orig_orchestrator
        else:
            app.dependency_overrides.pop(get_rag_orchestrator, None)
        if orig_settings:
            app.dependency_overrides[get_app_settings] = orig_settings
        else:
            app.dependency_overrides.pop(get_app_settings, None)


@pytest.mark.asyncio
async def test_query_stream_sse(async_client: AsyncClient) -> None:
    """Verify POST /api/v1/query/stream yields Server-Sent Events with token deltas and complete event."""

    async def mock_stream(**kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "token", "delta": "Kubernetes "}
        yield {"type": "token", "delta": "probes."}
        yield {
            "type": "complete",
            "answer": "Kubernetes probes.",
            "citations": [1],
            "confidence_score": 0.95,
            "grounded": True,
            "insufficient_context": False,
            "fallback_applied": False,
            "request_id": "stream-req-123",
            "citation_manifest": "References: None",
            "referenced_documents": [],
            "retrieval_metadata": {},
        }

    mock_orchestrator = MagicMock()
    mock_orchestrator.execute_stream = mock_stream

    app = async_client._transport.app  # type: ignore[attr-defined]
    app.dependency_overrides[get_rag_orchestrator] = lambda: mock_orchestrator

    try:
        response = await async_client.post(
            "/api/v1/query/stream",
            json={"query": "What are probes?"},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        body_text = response.text
        assert 'data: {"type": "token", "delta": "Kubernetes "}' in body_text
        assert 'data: {"type": "token", "delta": "probes."}' in body_text
        assert '"type": "complete"' in body_text
        assert '"stream-req-123"' in body_text
    finally:
        app.dependency_overrides.pop(get_rag_orchestrator, None)


@pytest.mark.asyncio
async def test_health_and_readiness_endpoints(async_client: AsyncClient) -> None:
    """Verify liveness and readiness probe endpoints."""
    # Root probes
    health_res = await async_client.get("/health")
    assert health_res.status_code == 200
    assert health_res.json()["status"] == "healthy"

    ready_res = await async_client.get("/ready")
    assert ready_res.status_code == 200
    assert ready_res.json()["status"] == "ready"
    assert "api" in ready_res.json()["checks"]

    # v1 probes
    v1_health = await async_client.get("/api/v1/health")
    assert v1_health.status_code == 200
    assert v1_health.json()["status"] == "healthy"

    v1_ready = await async_client.get("/api/v1/ready")
    assert v1_ready.status_code == 200
    assert v1_ready.json()["status"] == "ready"
