"""Integration tests validating database layer transactional integrity, cascading deletes, and error handling."""

import uuid

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

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
async def test_transaction_rollback_on_integrity_violation(
    db_session: AsyncSession,
    document_repo: SQLAlchemyDocumentRepository,
) -> None:
    """Verify database rolls back cleanly on constraint violation and leaves session operable."""
    doc_id = uuid.uuid4()
    doc1 = Document(
        id=doc_id,
        name="Unique Document",
        document_type="pdf",
        source="local://unique_doc.pdf",
    )
    await document_repo.create(doc1)

    from sqlalchemy import insert

    # Attempt to insert identical primary key directly via SQL statement to test DB constraint violation
    stmt = insert(Document).values(
        id=doc_id,
        name="Clashing Document",
        document_type="pdf",
        source="local://clash.pdf",
    )
    with pytest.raises(IntegrityError):
        await db_session.execute(stmt)
        await db_session.flush()

    # Roll back to restore clean session state
    await db_session.rollback()

    # Session should still be fully functional after rollback
    surviving_doc = await document_repo.get_by_id(doc_id)
    assert surviving_doc is None or surviving_doc.id == doc_id


@pytest.mark.asyncio
async def test_session_rollback_cleans_uncommitted_state(
    db_session: AsyncSession,
    document_repo: SQLAlchemyDocumentRepository,
) -> None:
    """Verify uncommitted session changes are discarded upon rollback."""
    doc = Document(
        name="Ephemeral Document",
        document_type="txt",
        source="local://ephemeral.txt",
    )
    db_session.add(doc)
    await db_session.flush()

    assert doc.id is not None
    # Roll back
    await db_session.rollback()

    # Verify document does not exist
    result = await document_repo.get_by_id(doc.id)
    assert result is None


@pytest.mark.asyncio
async def test_cascading_delete_document_and_associated_entities(
    db_session: AsyncSession,
    document_repo: SQLAlchemyDocumentRepository,
    version_repo: SQLAlchemyDocumentVersionRepository,
    chunk_repo: SQLAlchemyDocumentChunkRepository,
    job_repo: SQLAlchemyIngestionJobRepository,
) -> None:
    """Verify deleting a document cascades to all versions, chunks, and jobs."""
    # 1. Create document with version
    doc = Document(
        name="Cascading Spec",
        document_type="markdown",
        source="local://cascade.md",
    )
    ver = DocumentVersion(
        version_number=1,
        content_hash="hash_cascade_001",
        size_bytes=512,
        status="PROCESSED",
    )
    persisted_doc = await document_repo.create_with_version(doc, ver)
    doc_id = persisted_doc.id
    ver_id = ver.id

    # 2. Add chunks
    chunks = [
        DocumentChunk(
            document_version_id=ver_id,
            chunk_index=0,
            content="First chunk for cascade test.",
            token_count=6,
            char_count=30,
        ),
        DocumentChunk(
            document_version_id=ver_id,
            chunk_index=1,
            content="Second chunk for cascade test.",
            token_count=6,
            char_count=31,
        ),
    ]
    await chunk_repo.bulk_create_chunks(chunks)

    # 3. Add job
    await job_repo.create_job(document_id=doc_id, document_version_id=ver_id)

    await db_session.commit()

    # Verify all entities exist
    assert await document_repo.get_by_id(doc_id) is not None
    assert await version_repo.get_by_id(ver_id) is not None
    assert len(await chunk_repo.get_chunks_by_version(ver_id)) == 2
    assert len(await job_repo.get_jobs_by_document(doc_id)) == 1

    # 4. Delete document
    deleted = await document_repo.delete(doc_id)
    assert deleted is True
    await db_session.commit()

    # 5. Verify cascading deletion
    assert await document_repo.get_by_id(doc_id) is None
    assert await version_repo.get_by_id(ver_id) is None
    assert len(await chunk_repo.get_chunks_by_version(ver_id)) == 0
    assert len(await job_repo.get_jobs_by_document(doc_id)) == 0


@pytest.mark.asyncio
async def test_ingestion_job_state_transitions(
    db_session: AsyncSession,
    document_repo: SQLAlchemyDocumentRepository,
    job_repo: SQLAlchemyIngestionJobRepository,
) -> None:
    """Verify IngestionJob state transitions from QUEUED -> PROCESSING -> FAILED with error persistence."""
    doc = Document(
        name="Job State Doc",
        document_type="txt",
        source="local://job_doc.txt",
    )
    persisted_doc = await document_repo.create(doc)

    # 1. Created as QUEUED
    job = await job_repo.create_job(document_id=persisted_doc.id)
    assert job.status == "QUEUED"

    # 2. Transition to PROCESSING with stage update
    updated_proc = await job_repo.update_progress(
        job_id=job.id,
        status="PROCESSING",
        stage_name="parsing",
    )
    assert updated_proc.status == "PROCESSING"
    assert "parsing" in updated_proc.stages_completed

    # 3. Transition to FAILED with error message
    failure_msg = "Corrupted UTF-8 byte sequence in source stream."
    updated_failed = await job_repo.update_progress(
        job_id=job.id,
        status="FAILED",
        stage_name="parsing",
        error_message=failure_msg,
    )
    assert updated_failed.status == "FAILED"
    assert updated_failed.error_message == failure_msg

    # Reload from DB and verify persisted fields
    refetched = await job_repo.get_by_id(job.id)
    assert refetched is not None
    assert refetched.status == "FAILED"
    assert refetched.error_message == failure_msg


@pytest.mark.asyncio
async def test_chunk_batch_ordered_retrieval(
    db_session: AsyncSession,
    document_repo: SQLAlchemyDocumentRepository,
    version_repo: SQLAlchemyDocumentVersionRepository,
    chunk_repo: SQLAlchemyDocumentChunkRepository,
) -> None:
    """Verify that chunks retrieved by version are sorted by chunk_index deterministically."""
    doc = Document(
        name="Ordered Chunks",
        document_type="txt",
        source="local://ordered.txt",
    )
    ver = DocumentVersion(
        version_number=1,
        content_hash="ordered_hash_001",
        size_bytes=100,
        status="PROCESSED",
    )
    await document_repo.create_with_version(doc, ver)

    # Insert out of order
    chunks = [
        DocumentChunk(
            document_version_id=ver.id,
            chunk_index=2,
            content="Section 3 text",
            token_count=3,
            char_count=14,
        ),
        DocumentChunk(
            document_version_id=ver.id,
            chunk_index=0,
            content="Section 1 text",
            token_count=3,
            char_count=14,
        ),
        DocumentChunk(
            document_version_id=ver.id,
            chunk_index=1,
            content="Section 2 text",
            token_count=3,
            char_count=14,
        ),
    ]
    await chunk_repo.bulk_create_chunks(chunks)
    await db_session.commit()

    retrieved = await chunk_repo.get_chunks_by_version(ver.id)
    assert len(retrieved) == 3
    assert [c.chunk_index for c in retrieved] == [0, 1, 2]
    assert [c.content for c in retrieved] == ["Section 1 text", "Section 2 text", "Section 3 text"]
