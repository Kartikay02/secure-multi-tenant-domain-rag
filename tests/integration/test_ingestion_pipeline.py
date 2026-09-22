"""Integration tests for IngestionService pipeline."""

from pathlib import Path

import pytest

from app.core.exceptions import ParsingError
from app.rag.storage.local import LocalStorageService
from app.repositories import (
    SQLAlchemyDocumentRepository,
    SQLAlchemyDocumentVersionRepository,
    SQLAlchemyIngestionJobRepository,
)
from app.services.ingestion_service import IngestionService


@pytest.mark.asyncio
async def test_ingestion_service_markdown_lifecycle(
    document_repo: SQLAlchemyDocumentRepository,
    version_repo: SQLAlchemyDocumentVersionRepository,
    job_repo: SQLAlchemyIngestionJobRepository,
    tmp_path: Path,
) -> None:
    """Verify complete ingestion pipeline for a Markdown document."""
    storage = LocalStorageService(base_dir=tmp_path)
    service = IngestionService(
        document_repo=document_repo,
        version_repo=version_repo,
        job_repo=job_repo,
        storage_service=storage,
    )

    md_content = (
        b"# Onboarding Guide\nWelcome to the engineering team.\n\n## Tools\nGit, Python, Docker."
    )
    response = await service.ingest_file(
        file_bytes=md_content,
        filename="onboarding.md",
        custom_metadata={"department": "Engineering"},
    )

    assert response.status == "PARSED"
    assert response.is_duplicate is False
    assert response.extracted_title == "Onboarding Guide"
    assert response.word_count > 0
    assert response.storage_uri.startswith("file://")

    # Verify document in DB
    doc = await document_repo.get_by_id(response.document_id)
    assert doc is not None
    assert doc.name == "onboarding.md"

    # Verify version in DB
    ver = await version_repo.get_by_id(response.document_version_id)
    assert ver is not None
    assert ver.status == "PARSED"
    assert ver.content_hash == response.content_hash

    # Verify job in DB
    job = await job_repo.get_by_id(response.job_id)
    assert job is not None
    assert job.status == "COMPLETED"
    assert "normalization" in job.stages_completed


@pytest.mark.asyncio
async def test_ingestion_service_duplicate_detection(
    document_repo: SQLAlchemyDocumentRepository,
    version_repo: SQLAlchemyDocumentVersionRepository,
    job_repo: SQLAlchemyIngestionJobRepository,
    tmp_path: Path,
) -> None:
    """Verify uploading the identical file a second time flags it as duplicate and skips re-parsing."""
    storage = LocalStorageService(base_dir=tmp_path)
    service = IngestionService(
        document_repo=document_repo,
        version_repo=version_repo,
        job_repo=job_repo,
        storage_service=storage,
    )

    file_bytes = b"Static policy documentation content."
    filename = "policy.txt"

    # 1. Initial ingestion
    first_resp = await service.ingest_file(file_bytes=file_bytes, filename=filename)
    assert first_resp.is_duplicate is False

    # 2. Duplicate ingestion
    second_resp = await service.ingest_file(file_bytes=file_bytes, filename=filename)
    assert second_resp.is_duplicate is True
    assert second_resp.document_id == first_resp.document_id
    assert second_resp.document_version_id == first_resp.document_version_id
    assert second_resp.content_hash == first_resp.content_hash


@pytest.mark.asyncio
async def test_ingestion_service_malformed_file_marks_failed(
    document_repo: SQLAlchemyDocumentRepository,
    version_repo: SQLAlchemyDocumentVersionRepository,
    job_repo: SQLAlchemyIngestionJobRepository,
    tmp_path: Path,
) -> None:
    """Verify malformed file parsing failure updates job and version status to FAILED."""
    storage = LocalStorageService(base_dir=tmp_path)
    service = IngestionService(
        document_repo=document_repo,
        version_repo=version_repo,
        job_repo=job_repo,
        storage_service=storage,
    )

    # %PDF- signature passes magic-byte check, but subsequent stream is corrupt
    corrupt_pdf_bytes = b"%PDF-1.7\nCorrupted binary stream with no xref or trailer"
    with pytest.raises(ParsingError):
        await service.ingest_file(file_bytes=corrupt_pdf_bytes, filename="bad.pdf")

    # Verify a failed job was recorded for the document
    jobs = await job_repo.list(limit=5)
    assert len(jobs) >= 1
    failed_job = jobs[0]
    assert failed_job.status == "FAILED"
    assert failed_job.error_message is not None
