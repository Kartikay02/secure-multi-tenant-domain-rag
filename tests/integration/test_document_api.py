"""Integration tests for Document API HTTP endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_upload_document_workflow(async_client: AsyncClient) -> None:
    """Verify upload, duplicate check, detail retrieval, job inspection, and deletion."""
    file_content = b"# Architecture Design\nThis document describes the clean architecture."
    files = {"file": ("architecture.md", file_content, "text/markdown")}

    # 1. First upload -> 201 Created
    response = await async_client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 201

    data = response.json()
    assert data["filename"] == "architecture.md"
    assert data["status"] == "PARSED"
    assert data["is_duplicate"] is False
    assert "document_id" in data
    doc_id = data["document_id"]

    # 2. Re-upload identical file -> 200 OK with is_duplicate = True
    dup_response = await async_client.post(
        "/api/v1/documents/upload",
        files={"file": ("architecture.md", file_content, "text/markdown")},
    )
    assert dup_response.status_code == 200
    dup_data = dup_response.json()
    assert dup_data["is_duplicate"] is True
    assert dup_data["document_id"] == doc_id

    # 3. Retrieve document details -> 200 OK
    get_response = await async_client.get(f"/api/v1/documents/{doc_id}")
    assert get_response.status_code == 200
    doc_details = get_response.json()
    assert doc_details["id"] == doc_id
    assert len(doc_details["versions"]) == 1
    assert doc_details["versions"][0]["status"] == "PARSED"

    # 4. Retrieve document jobs -> 200 OK
    jobs_response = await async_client.get(f"/api/v1/documents/{doc_id}/jobs")
    assert jobs_response.status_code == 200
    jobs = jobs_response.json()
    assert len(jobs) >= 1
    assert jobs[0]["status"] == "COMPLETED"

    # 5. Delete document -> 204 No Content
    del_response = await async_client.delete(f"/api/v1/documents/{doc_id}")
    assert del_response.status_code == 204

    # 6. Verify 404 after deletion
    get_after_del = await async_client.get(f"/api/v1/documents/{doc_id}")
    assert get_after_del.status_code == 404


@pytest.mark.asyncio
async def test_upload_unsupported_file_type_returns_415(async_client: AsyncClient) -> None:
    """Verify uploading an unsupported file type returns HTTP 415."""
    files = {"file": ("malicious.exe", b"MZ\x90\x00executable binary", "application/x-msdownload")}
    response = await async_client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 415

    data = response.json()
    assert data["error_code"] == "UNSUPPORTED_FILE_TYPE"
    assert "exe" in data["detail"]


@pytest.mark.asyncio
async def test_upload_empty_file_returns_400(async_client: AsyncClient) -> None:
    """Verify uploading an empty file returns HTTP 400."""
    files = {"file": ("empty.txt", b"", "text/plain")}
    response = await async_client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 400

    data = response.json()
    assert data["error_code"] == "FILE_VALIDATION_ERROR"
    assert "empty" in data["detail"].lower()
