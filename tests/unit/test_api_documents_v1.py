"""Unit/integration tests for Document v1 API endpoints."""

import uuid

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_documents_lifecycle_and_pagination(async_client: AsyncClient) -> None:
    """Verify document upload, paginated listing, detail fetch, processing, and deletion."""
    # 1. Upload Document 1
    doc1_content = b"# Document One\nFirst document content for testing pagination."
    res1 = await async_client.post(
        "/api/v1/documents",
        files={"file": ("doc1.md", doc1_content, "text/markdown")},
    )
    assert res1.status_code == 201
    doc1_id = res1.json()["document_id"]

    # 2. Upload Document 2
    doc2_content = b"# Document Two\nSecond document content for testing pagination."
    res2 = await async_client.post(
        "/api/v1/documents",
        files={"file": ("doc2.md", doc2_content, "text/markdown")},
    )
    assert res2.status_code == 201
    doc2_id = res2.json()["document_id"]

    # 3. List Documents with pagination: page=1, page_size=1
    page1_res = await async_client.get("/api/v1/documents?page=1&page_size=1")
    assert page1_res.status_code == 200
    page1_data = page1_res.json()
    assert page1_data["total"] >= 2
    assert page1_data["page"] == 1
    assert page1_data["page_size"] == 1
    assert len(page1_data["items"]) == 1

    # 4. List Documents with pagination: page=2, page_size=1
    page2_res = await async_client.get("/api/v1/documents?page=2&page_size=1")
    assert page2_res.status_code == 200
    page2_data = page2_res.json()
    assert len(page2_data["items"]) == 1
    # Items across page 1 and page 2 should be different
    assert page1_data["items"][0]["id"] != page2_data["items"][0]["id"]

    # 5. Process Document 1 (trigger chunking and embedding)
    proc_res = await async_client.post(f"/api/v1/documents/{doc1_id}/process")
    assert proc_res.status_code == 200
    proc_data = proc_res.json()
    assert proc_data["document_id"] == doc1_id
    assert proc_data["status"] == "EMBEDDED"
    assert proc_data["total_chunks"] >= 1
    assert proc_data["total_vectors"] >= 1

    # 6. Fetch Document Details
    detail_res = await async_client.get(f"/api/v1/documents/{doc1_id}")
    assert detail_res.status_code == 200
    detail_data = detail_res.json()
    assert detail_data["id"] == doc1_id
    assert detail_data["name"] == "doc1.md"
    assert len(detail_data["versions"]) >= 1

    # 7. Process Non-Existent Document returns 404
    non_existent_id = uuid.uuid4()
    bad_proc = await async_client.post(f"/api/v1/documents/{non_existent_id}/process")
    assert bad_proc.status_code == 404
    assert bad_proc.json()["error_code"] == "NOT_FOUND"

    # 8. Delete Document
    del_res = await async_client.delete(f"/api/v1/documents/{doc1_id}")
    assert del_res.status_code == 204

    # 9. Verify 404 after deletion
    del_verify = await async_client.get(f"/api/v1/documents/{doc1_id}")
    assert del_verify.status_code == 404

    # Clean up doc2
    await async_client.delete(f"/api/v1/documents/{doc2_id}")
