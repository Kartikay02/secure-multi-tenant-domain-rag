"""Integration tests for EmbeddingService and end-to-end embedding pipeline."""

from pathlib import Path

import pytest

from app.rag.chunking.recursive import RecursiveTokenChunker
from app.rag.embeddings.mock import MockEmbeddingProvider
from app.rag.embeddings.vector_mock import InMemoryVectorStore
from app.rag.storage.local import LocalStorageService
from app.repositories import (
    SQLAlchemyDocumentChunkRepository,
    SQLAlchemyDocumentRepository,
    SQLAlchemyDocumentVersionRepository,
    SQLAlchemyIngestionJobRepository,
)
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.ingestion_service import IngestionService


@pytest.mark.asyncio
async def test_embedding_pipeline_and_persistence(
    document_repo: SQLAlchemyDocumentRepository,
    version_repo: SQLAlchemyDocumentVersionRepository,
    chunk_repo: SQLAlchemyDocumentChunkRepository,
    job_repo: SQLAlchemyIngestionJobRepository,
    tmp_path: Path,
) -> None:
    """Verify end-to-end chunking to vector embedding generation and persistence."""
    # 1. Ingest document
    storage = LocalStorageService(base_dir=tmp_path)
    ingest_service = IngestionService(
        document_repo=document_repo,
        version_repo=version_repo,
        job_repo=job_repo,
        storage_service=storage,
    )

    doc_text = b"""# Machine Learning Systems
Welcome to our production ML guidelines.

## Vector Search Architectures
Modern RAG systems utilize dense vector embeddings for semantic document retrieval.
Embeddings map unstructured text into continuous high-dimensional vector spaces.
Cosine similarity or inner products measure geometric proximity between queries and chunks.
"""

    ingest_resp = await ingest_service.ingest_file(
        file_bytes=doc_text,
        filename="ml_systems.md",
    )
    assert ingest_resp.status == "PARSED"

    # 2. Chunk document
    chunker = RecursiveTokenChunker(chunk_size=25, chunk_overlap=5)
    chunking_service = ChunkingService(
        document_repo=document_repo,
        version_repo=version_repo,
        chunk_repo=chunk_repo,
        job_repo=job_repo,
        chunker=chunker,
    )

    chunks = await chunking_service.chunk_document_version(
        document_id=ingest_resp.document_id,
        version_id=ingest_resp.document_version_id,
    )
    assert len(chunks) > 1

    # 3. Embed document version
    embedding_provider = MockEmbeddingProvider(dimension=384, model_name="bge-small-en-v1.5")
    vector_store = InMemoryVectorStore()
    embedding_service = EmbeddingService(
        document_repo=document_repo,
        version_repo=version_repo,
        chunk_repo=chunk_repo,
        job_repo=job_repo,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    vector_records = await embedding_service.embed_document_version(
        document_id=ingest_resp.document_id,
        version_id=ingest_resp.document_version_id,
    )

    # 4. Verify vector records
    assert len(vector_records) == len(chunks)
    for rec in vector_records:
        assert len(rec.vector) == 384
        assert rec.document_id == str(ingest_resp.document_id)
        assert rec.version_id == str(ingest_resp.document_version_id)
        assert "chunk_index" in rec.metadata
        assert "document_name" in rec.metadata

    # 5. Verify vector store persistence
    assert await vector_store.count() == len(chunks)
    first_record = await vector_store.get(str(chunks[0].id))
    assert first_record is not None
    assert first_record.vector == vector_records[0].vector

    # 6. Verify database updates
    ver = await version_repo.get_by_id(ingest_resp.document_version_id)
    assert ver is not None
    assert ver.status == "EMBEDDED"

    db_chunks = await chunk_repo.get_chunks_by_version(ingest_resp.document_version_id)
    for chunk in db_chunks:
        assert chunk.embedding_id is not None
        assert chunk.embedding_id == str(chunk.id)

    # 7. Verify ingestion job state
    jobs = await job_repo.get_jobs_by_document(ingest_resp.document_id)
    assert len(jobs) >= 1
    latest_job = jobs[0]
    assert latest_job.status == "COMPLETED"
    assert "embedding" in latest_job.stages_completed
    assert latest_job.stats["dimension"] == 384
    assert latest_job.stats["total_vectors"] == len(chunks)

    # 8. Test Idempotency: Re-embed same version
    re_embedded = await embedding_service.embed_document_version(
        document_id=ingest_resp.document_id,
        version_id=ingest_resp.document_version_id,
    )
    assert len(re_embedded) == len(chunks)
    # Total in vector store remains unchanged (upsert)
    assert await vector_store.count() == len(chunks)
