"""End-to-end integration tests for production FastAPI v1 APIs."""

import pytest
from httpx import AsyncClient

from app.api.deps import get_rate_limiter
from app.api.rate_limiting import InMemoryRateLimiter


@pytest.mark.asyncio
async def test_full_rag_api_lifecycle(async_client: AsyncClient) -> None:
    """Execute full end-to-end HTTP API lifecycle:

    Health -> Upload -> Process -> List -> Query -> Stream Query -> Auth Check -> Rate Limit -> Delete.
    """
    # 1. Health & Readiness Probes
    health_resp = await async_client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "healthy"

    ready_resp = await async_client.get("/ready")
    assert ready_resp.status_code == 200
    assert ready_resp.json()["status"] == "ready"

    # 2. Upload Document
    doc_content = b"""# Production RAG System Operations Manual

## Section 1: Ingestion and Parsing
The ingestion subsystem handles multi-format file uploads including Markdown, PDF, and DOCX documents.
Document content hashes prevent redundant index operations and eliminate duplicate chunk storage.

## Section 2: Hybrid Retrieval and Reranking
In hybrid retrieval, dense and lexical rankings are combined using HNSW vector search and lexical BM25 rankings via Reciprocal Rank Fusion.
Cross-encoder rerankers calibrate relevance scores before context window packing.
"""

    upload_resp = await async_client.post(
        "/api/v1/documents",
        files={"file": ("operations_manual.md", doc_content, "text/markdown")},
    )
    assert upload_resp.status_code == 201
    upload_data = upload_resp.json()
    doc_id = upload_data["document_id"]
    assert upload_data["filename"] == "operations_manual.md"
    assert upload_data["status"] == "PARSED"

    # 3. Process Document (Chunking + Vector Indexing)
    proc_resp = await async_client.post(f"/api/v1/documents/{doc_id}/process")
    assert proc_resp.status_code == 200
    proc_data = proc_resp.json()
    assert proc_data["document_id"] == doc_id
    assert proc_data["status"] == "EMBEDDED"
    assert proc_data["total_chunks"] >= 2
    assert proc_data["total_vectors"] >= 2

    # 4. List Documents with Pagination
    list_resp = await async_client.get("/api/v1/documents?page=1&page_size=10")
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert list_data["total"] >= 1
    assert any(item["id"] == doc_id for item in list_data["items"])

    # 5. Execute Grounded RAG Query
    query_resp = await async_client.post(
        "/api/v1/query",
        json={
            "query": "How are dense and lexical rankings combined in hybrid retrieval?",
            "top_k": 3,
            "temperature": 0.0,
        },
    )
    assert query_resp.status_code == 200
    query_data = query_resp.json()
    assert query_data["answer"] != ""
    assert query_data["grounded"] is True
    assert query_data["confidence_score"] > 0.0
    assert "request_id" in query_data
    assert "retrieval_metadata" in query_data

    # 6. Execute RAG Query via SSE Stream
    stream_resp = await async_client.post(
        "/api/v1/query/stream",
        json={"query": "Explain how document content hashes prevent redundant index operations"},
    )
    assert stream_resp.status_code == 200
    assert "text/event-stream" in stream_resp.headers["content-type"]
    stream_text = stream_resp.text
    assert "data: {" in stream_text
    assert '"type": "complete"' in stream_text

    # 7. Verify API Authentication Rejection with Invalid Key
    unauth_resp = await async_client.get(
        "/api/v1/documents",
        headers={"X-API-Key": "completely-invalid-key"},
    )
    assert unauth_resp.status_code == 401
    assert unauth_resp.json()["error_code"] == "UNAUTHORIZED"

    # 8. Verify Rate Limiting Enforcement
    # Override rate limiter with limit=2 for this specific test
    app = async_client._transport.app  # type: ignore[attr-defined]
    tight_limiter = InMemoryRateLimiter(max_requests=2, window_seconds=60)
    app.dependency_overrides[get_rate_limiter] = lambda: tight_limiter

    try:
        # Request 1 & 2: allowed
        r1 = await async_client.get("/api/v1/documents")
        assert r1.status_code == 200
        r2 = await async_client.get("/api/v1/documents")
        assert r2.status_code == 200

        # Request 3: rate limited
        r3 = await async_client.get("/api/v1/documents")
        assert r3.status_code == 429
        assert r3.json()["error_code"] == "RATE_LIMIT_EXCEEDED"
    finally:
        app.dependency_overrides.pop(get_rate_limiter, None)

    # 9. Clean up Document
    del_resp = await async_client.delete(f"/api/v1/documents/{doc_id}")
    assert del_resp.status_code == 204

    # Verify 404
    get_del = await async_client.get(f"/api/v1/documents/{doc_id}")
    assert get_del.status_code == 404
