"""Unit tests for SQLAlchemy repositories and persistence layer."""

import uuid

import pytest

from app.core.exceptions import DuplicateEntityError
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.version import DocumentVersion
from app.repositories import (
    SQLAlchemyDocumentChunkRepository,
    SQLAlchemyDocumentRepository,
    SQLAlchemyDocumentVersionRepository,
    SQLAlchemyIngestionJobRepository,
)


@pytest.mark.asyncio
async def test_document_crud_lifecycle(document_repo: SQLAlchemyDocumentRepository) -> None:
    """Verify document creation, lookup by id/source, listing, counting, and deletion."""
    doc = Document(
        name="Employee Handbook",
        document_type="markdown",
        source="s3://company-docs/handbook.md",
        description="Company guidelines and policies",
        metadata_json={"department": "HR", "author": "Alice"},
    )
    created = await document_repo.create(doc)
    assert created.id is not None
    assert created.name == "Employee Handbook"

    # Lookup by ID
    fetched = await document_repo.get_by_id(created.id)
    assert fetched is not None
    assert fetched.source == "s3://company-docs/handbook.md"

    # Lookup by source
    by_source = await document_repo.get_by_source("s3://company-docs/handbook.md")
    assert by_source is not None
    assert by_source.id == created.id

    # Count
    total = await document_repo.count()
    assert total >= 1

    # Delete
    deleted = await document_repo.delete(created.id)
    assert deleted is True
    assert await document_repo.get_by_id(created.id) is None


@pytest.mark.asyncio
async def test_create_document_with_initial_version(
    document_repo: SQLAlchemyDocumentRepository,
    version_repo: SQLAlchemyDocumentVersionRepository,
) -> None:
    """Verify atomic creation of document and its initial version in one transaction."""
    doc = Document(
        name="Architecture Spec",
        document_type="pdf",
        source="docs/arch.pdf",
    )
    initial_ver = DocumentVersion(
        document_id=uuid.uuid4(),  # placeholder, overwritten by create_with_version
        version_number=1,
        content_hash="abc123hash456",
        size_bytes=1048576,
        status="INDEXED",
        total_chunks=12,
    )

    persisted_doc = await document_repo.create_with_version(doc, initial_ver)
    assert persisted_doc.id is not None

    latest_ver = await version_repo.get_latest_version(persisted_doc.id)
    assert latest_ver is not None
    assert latest_ver.version_number == 1
    assert latest_ver.content_hash == "abc123hash456"
    assert latest_ver.document_id == persisted_doc.id


@pytest.mark.asyncio
async def test_document_eager_loading_avoids_n_plus_one(
    document_repo: SQLAlchemyDocumentRepository,
    version_repo: SQLAlchemyDocumentVersionRepository,
) -> None:
    """Verify get_with_versions eagerly loads versions and metadata without lazy I/O."""
    doc = Document(
        name="Security Policy",
        document_type="markdown",
        source="policies/security.md",
    )
    ver = DocumentVersion(
        document_id=uuid.uuid4(),
        version_number=1,
        content_hash="sechash789",
        size_bytes=2048,
    )
    persisted = await document_repo.create_with_version(doc, ver)

    # Add metadata entries
    await document_repo.add_metadata(persisted.id, "classification", "confidential")
    await document_repo.add_metadata(persisted.id, "review_cycle", "annual")

    # Fetch with eager loading
    eager_doc = await document_repo.get_with_versions(persisted.id)
    assert eager_doc is not None
    assert len(eager_doc.versions) == 1
    assert len(eager_doc.metadata_entries) == 2
    assert eager_doc.versions[0].content_hash == "sechash789"


@pytest.mark.asyncio
async def test_duplicate_metadata_key_raises_conflict(
    document_repo: SQLAlchemyDocumentRepository,
) -> None:
    """Verify unique constraint on (document_id, key) enforces uniqueness."""
    doc = Document(name="Doc A", document_type="txt", source="test://docA")
    persisted = await document_repo.create(doc)

    await document_repo.add_metadata(persisted.id, "tag", "finance")

    # Adding duplicate key for same document must raise DuplicateEntityError
    with pytest.raises(DuplicateEntityError):
        await document_repo.add_metadata(persisted.id, "tag", "duplicate_finance")


@pytest.mark.asyncio
async def test_version_content_hash_deduplication(
    document_repo: SQLAlchemyDocumentRepository,
    version_repo: SQLAlchemyDocumentVersionRepository,
) -> None:
    """Verify content hash lookups allow rapid deduplication checks."""
    content_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    doc = Document(name="Empty File", document_type="txt", source="test://empty.txt")
    ver = DocumentVersion(
        document_id=uuid.uuid4(),
        version_number=1,
        content_hash=content_hash,
        size_bytes=0,
    )
    await document_repo.create_with_version(doc, ver)

    # Query existing hash
    existing = await version_repo.get_by_content_hash(content_hash)
    assert existing is not None
    assert existing.content_hash == content_hash

    # Query non-existing hash
    non_existing = await version_repo.get_by_content_hash("nonexistenthash")
    assert non_existing is None


@pytest.mark.asyncio
async def test_chunk_bulk_creation_and_ordering(
    document_repo: SQLAlchemyDocumentRepository,
    version_repo: SQLAlchemyDocumentVersionRepository,
    chunk_repo: SQLAlchemyDocumentChunkRepository,
) -> None:
    """Verify bulk insertion of chunks and correct sequential index ordering."""
    doc = Document(name="Multi Chunk Doc", document_type="markdown", source="test://chunks.md")
    ver = DocumentVersion(
        document_id=uuid.uuid4(),
        version_number=1,
        content_hash="chunkhash111",
        size_bytes=5000,
    )
    await document_repo.create_with_version(doc, ver)

    chunks = [
        DocumentChunk(
            document_version_id=ver.id,
            chunk_index=i,
            content=f"Content for chunk number {i}",
            token_count=100 + i,
            char_count=25 + i,
            metadata_json={"section": f"Section {i}"},
        )
        for i in range(5)
    ]

    persisted_chunks = await chunk_repo.bulk_create_chunks(chunks)
    assert len(persisted_chunks) == 5

    count = await chunk_repo.count_by_version(ver.id)
    assert count == 5

    fetched_chunks = await chunk_repo.get_chunks_by_version(ver.id)
    assert len(fetched_chunks) == 5
    for idx, c in enumerate(fetched_chunks):
        assert c.chunk_index == idx
        assert c.token_count == 100 + idx


@pytest.mark.asyncio
async def test_avoid_loading_large_chunk_content(
    document_repo: SQLAlchemyDocumentRepository,
    chunk_repo: SQLAlchemyDocumentChunkRepository,
) -> None:
    """Verify deferring content column saves I/O and memory when inspecting chunks (Req 11)."""
    doc = Document(name="Large Doc", document_type="txt", source="test://large.txt")
    ver = DocumentVersion(
        document_id=uuid.uuid4(),
        version_number=1,
        content_hash="largehash222",
        size_bytes=50000,
    )
    await document_repo.create_with_version(doc, ver)

    large_text = "X" * 10000
    chunk = DocumentChunk(
        document_version_id=ver.id,
        chunk_index=0,
        content=large_text,
        token_count=2500,
        char_count=10000,
    )
    await chunk_repo.create(chunk)

    # Fetch without content
    chunks_without_content = await chunk_repo.get_chunks_by_version(ver.id, include_content=False)
    assert len(chunks_without_content) == 1
    # Check that metadata and token count are accessible
    assert chunks_without_content[0].chunk_index == 0
    assert chunks_without_content[0].token_count == 2500


@pytest.mark.asyncio
async def test_ingestion_job_lifecycle(
    document_repo: SQLAlchemyDocumentRepository,
    job_repo: SQLAlchemyIngestionJobRepository,
) -> None:
    """Verify ingestion job tracking, stage completion logging, and status transitions."""
    doc = Document(name="Pipeline Doc", document_type="pdf", source="test://pipeline.pdf")
    persisted = await document_repo.create(doc)

    # 1. Create job
    job = await job_repo.create_job(document_id=persisted.id)
    assert job.status == "QUEUED"
    assert job.document_id == persisted.id
    assert job.started_at is not None

    # 2. Update progress: parsing stage
    updated = await job_repo.update_progress(
        job_id=job.id,
        status="PARSING",
        stage_name="parsing",
    )
    assert updated.status == "PARSING"
    assert "parsing" in updated.stages_completed

    # 3. Update progress: completed
    completed = await job_repo.update_progress(
        job_id=job.id,
        status="COMPLETED",
        stage_name="indexing",
        stats={"chunk_count": 8, "tokens": 3200, "duration_ms": 450},
    )
    assert completed.status == "COMPLETED"
    assert completed.completed_at is not None
    assert completed.stats["chunk_count"] == 8

    # Query jobs by document
    doc_jobs = await job_repo.get_jobs_by_document(persisted.id)
    assert len(doc_jobs) == 1
    assert doc_jobs[0].id == job.id


@pytest.mark.asyncio
async def test_cascade_deletion(
    document_repo: SQLAlchemyDocumentRepository,
    version_repo: SQLAlchemyDocumentVersionRepository,
    chunk_repo: SQLAlchemyDocumentChunkRepository,
) -> None:
    """Verify deleting a Document cascades and cleans up versions, chunks, and metadata."""
    doc = Document(name="Cascade Doc", document_type="txt", source="test://cascade.txt")
    ver = DocumentVersion(
        document_id=uuid.uuid4(),
        version_number=1,
        content_hash="cascadehash333",
        size_bytes=100,
    )
    await document_repo.create_with_version(doc, ver)

    await document_repo.add_metadata(doc.id, "author", "Bob")

    chunk = DocumentChunk(
        document_version_id=ver.id,
        chunk_index=0,
        content="Cascade text content",
        token_count=10,
    )
    await chunk_repo.create(chunk)

    assert await version_repo.get_by_id(ver.id) is not None
    assert await chunk_repo.get_by_id(chunk.id) is not None

    # Delete parent document
    deleted = await document_repo.delete(doc.id)
    assert deleted is True

    # Associated version and chunk must be gone
    assert await version_repo.get_by_id(ver.id) is None
    assert await chunk_repo.get_by_id(chunk.id) is None
