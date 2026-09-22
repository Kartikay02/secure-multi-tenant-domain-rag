"""Integration tests for VectorRetrievalService and end-to-end similarity search."""

from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.chunking.recursive import RecursiveTokenChunker
from app.rag.embeddings.mock import MockEmbeddingProvider
from app.rag.embeddings.vector_interfaces import VectorRecord
from app.rag.storage.local import LocalStorageService
from app.rag.vector.pgvector import PGVectorStore
from app.repositories import (
    SQLAlchemyDocumentChunkRepository,
    SQLAlchemyDocumentRepository,
    SQLAlchemyDocumentVersionRepository,
    SQLAlchemyIngestionJobRepository,
)
from app.services.chunking_service import ChunkingService
from app.services.ingestion_service import IngestionService
from app.services.retrieval_service import VectorRetrievalService


@pytest.mark.asyncio
async def test_end_to_end_vector_retrieval(
    db_session: AsyncSession,
    document_repo: SQLAlchemyDocumentRepository,
    version_repo: SQLAlchemyDocumentVersionRepository,
    chunk_repo: SQLAlchemyDocumentChunkRepository,
    job_repo: SQLAlchemyIngestionJobRepository,
    tmp_path: Path,
) -> None:
    """Verify ingestion -> chunking -> embedding -> pgvector indexing -> similarity search."""
    # 1. Ingest document with distinct topical sections
    storage = LocalStorageService(base_dir=tmp_path)
    ingest_service = IngestionService(
        document_repo=document_repo,
        version_repo=version_repo,
        job_repo=job_repo,
        storage_service=storage,
    )

    doc_text = b"""# Technical Architecture Manual

## Database Engineering
PostgreSQL is our primary relational persistence engine.
We utilize transactions with ACID guarantees, WAL archiving, and connection pooling.
Foreign keys enforce relational integrity across all document entities.

## Frontend Stylesheet Standards
Modern Web UI components rely on Tailwind CSS for utility-first styling.
Buttons and modals adhere to accessible design guidelines and high contrast color themes.
"""

    ingest_res = await ingest_service.ingest_file(
        file_bytes=doc_text,
        filename="architecture_manual.md",
    )
    assert ingest_res.status == "PARSED"

    # 2. Chunk document by section
    chunker = RecursiveTokenChunker(chunk_size=40, chunk_overlap=5)
    chunking_service = ChunkingService(
        document_repo=document_repo,
        version_repo=version_repo,
        chunk_repo=chunk_repo,
        job_repo=job_repo,
        chunker=chunker,
    )
    chunks = await chunking_service.chunk_document_version(
        document_id=ingest_res.document_id,
        version_id=ingest_res.document_version_id,
    )
    assert len(chunks) >= 2

    # 3. Generate embeddings with deterministic mock provider
    embedding_provider = MockEmbeddingProvider(dimension=128, model_name="test-bge")
    texts = [c.content for c in chunks]
    vectors = await embedding_provider.embed_documents(texts)

    # 4. Index vectors into PGVectorStore
    vector_store = PGVectorStore(session=db_session, dimension=128)
    vector_records = [
        VectorRecord(
            id=str(c.id),
            vector=v,
            document_id=str(ingest_res.document_id),
            version_id=str(ingest_res.document_version_id),
            metadata=c.metadata_json,
        )
        for c, v in zip(chunks, vectors, strict=True)
    ]
    upserted = await vector_store.upsert_vectors(vector_records)
    assert upserted == len(chunks)

    # 5. Initialize VectorRetrievalService
    retrieval_service = VectorRetrievalService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        default_top_k=2,
    )

    # 6. Query matching Database section exactly
    # Since MockEmbeddingProvider generates identical vectors for identical text,
    # searching with chunk[0]'s text will yield an exact score of 1.0!
    db_query = chunks[0].content
    db_results = await retrieval_service.search(query=db_query, top_k=2)

    assert len(db_results) > 0
    top_db = db_results[0]
    assert top_db.chunk_id == chunks[0].id
    assert top_db.document_id == ingest_res.document_id
    assert "Database" in top_db.text or "PostgreSQL" in top_db.text
    assert top_db.score == pytest.approx(1.0, rel=1e-2)
    assert "document_name" in top_db.metadata

    # 7. Query matching Frontend section exactly
    frontend_chunk = next(c for c in chunks if "Frontend" in c.content or "Tailwind" in c.content)
    fe_results = await retrieval_service.search(query=frontend_chunk.content, top_k=2)

    assert len(fe_results) > 0
    top_fe = fe_results[0]
    assert top_fe.chunk_id == frontend_chunk.id
    assert "Frontend" in top_fe.text or "Tailwind" in top_fe.text
    assert top_fe.score == pytest.approx(1.0, rel=1e-2)

    # 8. Empty query test
    assert await retrieval_service.search("") == []
    assert await retrieval_service.search("   ") == []
