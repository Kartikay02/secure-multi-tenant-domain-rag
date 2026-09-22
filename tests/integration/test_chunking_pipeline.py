"""Integration tests for ChunkingService and database persistence."""

from pathlib import Path

import pytest

from app.rag.chunking.recursive import RecursiveTokenChunker
from app.rag.storage.local import LocalStorageService
from app.repositories import (
    SQLAlchemyDocumentChunkRepository,
    SQLAlchemyDocumentRepository,
    SQLAlchemyDocumentVersionRepository,
    SQLAlchemyIngestionJobRepository,
)
from app.services.chunking_service import ChunkingService
from app.services.ingestion_service import IngestionService


@pytest.mark.asyncio
async def test_chunking_service_pipeline_and_idempotency(
    document_repo: SQLAlchemyDocumentRepository,
    version_repo: SQLAlchemyDocumentVersionRepository,
    chunk_repo: SQLAlchemyDocumentChunkRepository,
    job_repo: SQLAlchemyIngestionJobRepository,
    tmp_path: Path,
) -> None:
    """Verify chunk generation, database persistence, and re-processing idempotency."""
    storage = LocalStorageService(base_dir=tmp_path)
    ingest_service = IngestionService(
        document_repo=document_repo,
        version_repo=version_repo,
        job_repo=job_repo,
        storage_service=storage,
    )

    doc_text = b"""# Engineering Handbook
Welcome to our platform.

## Architecture Guidelines
We adhere strictly to clean architecture and hexagonal ports and adapters.
Every external service is hidden behind an abstract interface protocol.

## Database Standards
PostgreSQL is the single source of truth for both metadata and vector embeddings.
Foreign keys must specify ON DELETE CASCADE and indexes must be added to all foreign keys.
"""

    ingest_resp = await ingest_service.ingest_file(
        file_bytes=doc_text,
        filename="handbook.md",
    )
    assert ingest_resp.status == "PARSED"

    # 1. Initialize ChunkingService
    chunker = RecursiveTokenChunker(chunk_size=30, chunk_overlap=5)
    chunking_service = ChunkingService(
        document_repo=document_repo,
        version_repo=version_repo,
        chunk_repo=chunk_repo,
        job_repo=job_repo,
        chunker=chunker,
    )

    # 2. Execute chunking
    persisted_chunks = await chunking_service.chunk_document_version(
        document_id=ingest_resp.document_id,
        version_id=ingest_resp.document_version_id,
    )

    assert len(persisted_chunks) > 1
    assert persisted_chunks[0].chunk_index == 0
    assert persisted_chunks[1].chunk_index == 1

    # Verify version state updated in DB
    ver = await version_repo.get_by_id(ingest_resp.document_version_id)
    assert ver is not None
    assert ver.status == "CHUNKED"
    assert ver.total_chunks == len(persisted_chunks)

    # Verify chunks present in DB
    db_chunks = await chunk_repo.get_chunks_by_version(ingest_resp.document_version_id)
    assert len(db_chunks) == len(persisted_chunks)

    # 3. Test Idempotency: Re-chunk the same version
    rechunked = await chunking_service.chunk_document_version(
        document_id=ingest_resp.document_id,
        version_id=ingest_resp.document_version_id,
    )

    # Total chunks in DB must still equal original count (zero duplicates created)
    total_in_db = await chunk_repo.count_by_version(ingest_resp.document_version_id)
    assert total_in_db == len(rechunked)
    assert len(rechunked) == len(persisted_chunks)
