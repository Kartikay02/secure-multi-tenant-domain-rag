"""Integration tests for HybridRetriever combining dense and lexical search."""

from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.chunking.recursive import RecursiveTokenChunker
from app.rag.embeddings.mock import MockEmbeddingProvider
from app.rag.embeddings.vector_interfaces import VectorRecord
from app.rag.retrieval.dense import DenseRetriever
from app.rag.retrieval.fusion import LinearCombinationFusion, ReciprocalRankFusion
from app.rag.retrieval.hybrid import HybridRetriever
from app.rag.retrieval.lexical import PGLexicalRetriever
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


@pytest.mark.asyncio
async def test_end_to_end_hybrid_retrieval_and_fusion(
    db_session: AsyncSession,
    document_repo: SQLAlchemyDocumentRepository,
    version_repo: SQLAlchemyDocumentVersionRepository,
    chunk_repo: SQLAlchemyDocumentChunkRepository,
    job_repo: SQLAlchemyIngestionJobRepository,
    tmp_path: Path,
) -> None:
    """Verify hybrid retrieval fuses dense and lexical results with agreement ranking."""
    # 1. Ingest document
    storage = LocalStorageService(base_dir=tmp_path)
    ingest_service = IngestionService(
        document_repo=document_repo,
        version_repo=version_repo,
        job_repo=job_repo,
        storage_service=storage,
    )

    doc_text = b"""# Infrastructure Operations Manual

## Section A: Relational Concurrency
Database engines handle parallel query workers through multi-version concurrency control.
Read locks and write barriers maintain snapshot isolation across transaction boundaries.

## Section B: Streaming Ingestion
KAFKA_BROKER_TIMEOUT_MS specifies maximum cluster consumer keepalive duration in milliseconds.

## Section C: Database Replication
PostgreSQL replication uses streaming WAL records to maintain hot standby database servers.
Replication lag is monitored through write-ahead log replay byte offsets.
"""

    ingest_res = await ingest_service.ingest_file(
        file_bytes=doc_text,
        filename="infra_manual.md",
    )
    assert ingest_res.status == "PARSED"

    # 2. Chunk document
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
    assert len(chunks) >= 3

    # 3. Generate embeddings & index into PGVectorStore
    embedding_provider = MockEmbeddingProvider(dimension=128, model_name="bge-mock")
    vector_store = PGVectorStore(session=db_session, dimension=128)

    texts = [c.content for c in chunks]
    vectors = await embedding_provider.embed_documents(texts)
    records = [
        VectorRecord(
            id=str(c.id),
            vector=v,
            document_id=str(ingest_res.document_id),
            version_id=str(ingest_res.document_version_id),
            metadata=c.metadata_json,
        )
        for c, v in zip(chunks, vectors, strict=True)
    ]
    await vector_store.upsert_vectors(records)

    # 4. Build retrievers and hybrid orchestrator
    dense_retriever = DenseRetriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )
    lexical_retriever = PGLexicalRetriever(session=db_session)

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        lexical_retriever=lexical_retriever,
        fusion_strategy=ReciprocalRankFusion(k=60),
        dense_top_k=5,
        sparse_top_k=5,
        final_top_k=3,
    )

    # 5. Query combining keywords and concepts from Section C
    # "PostgreSQL replication streaming WAL"
    query = "PostgreSQL replication streaming WAL"
    results = await hybrid_retriever.retrieve(query=query, top_k=3)

    assert len(results) >= 2

    # Top chunk should be Section C due to strong mutual agreement in both dense and lexical
    top_chunk = results[0]
    assert "Section C" in top_chunk.text or "replication" in top_chunk.text.lower()
    # Verify score preservation
    assert top_chunk.sparse_score is not None
    assert top_chunk.dense_score is not None
    assert top_chunk.metadata["fusion_method"] == "rrf"

    # 6. Test Replaceable Fusion Strategy (LinearCombinationFusion)
    linear_hybrid = HybridRetriever(
        dense_retriever=dense_retriever,
        lexical_retriever=lexical_retriever,
        fusion_strategy=LinearCombinationFusion(dense_weight=0.6, sparse_weight=0.4),
        final_top_k=3,
    )

    linear_results = await linear_hybrid.retrieve(query=query, top_k=3)
    assert len(linear_results) >= 2
    assert linear_results[0].metadata["fusion_method"] == "linear"
    assert 0.0 <= linear_results[0].score <= 1.0
