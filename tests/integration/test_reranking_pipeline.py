"""Integration tests for RerankingPipeline end-to-end with database and hybrid retrieval."""

from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.chunking.recursive import RecursiveTokenChunker
from app.rag.embeddings.mock import MockEmbeddingProvider
from app.rag.embeddings.vector_interfaces import VectorRecord
from app.rag.reranking.mock import MockReranker
from app.rag.reranking.pipeline import RerankingPipeline
from app.rag.retrieval.dense import DenseRetriever
from app.rag.retrieval.fusion import ReciprocalRankFusion
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
async def test_end_to_end_reranking_pipeline_with_fallback(
    db_session: AsyncSession,
    document_repo: SQLAlchemyDocumentRepository,
    version_repo: SQLAlchemyDocumentVersionRepository,
    chunk_repo: SQLAlchemyDocumentChunkRepository,
    job_repo: SQLAlchemyIngestionJobRepository,
    tmp_path: Path,
) -> None:
    """Verify end-to-end workflow: Ingest -> Chunk -> Index -> Hybrid Search -> Cross-Encoder Rerank -> Graceful Fallback."""
    # 1. Ingest document
    storage = LocalStorageService(base_dir=tmp_path)
    ingest_service = IngestionService(
        document_repo=document_repo,
        version_repo=version_repo,
        job_repo=job_repo,
        storage_service=storage,
    )

    doc_text = b"""# High Availability Architecture Guide

## Section A: Distributed Consensus and Raft
Raft achieves consensus by electing a distinguished leader which takes responsibility
for log replication and commit index broadcast to cluster followers.

## Section B: Event Driven Microservices
Kafka topic partition leaders manage write commits while consumer groups track offset commits.

## Section C: PostgreSQL Physical Streaming Replication
PostgreSQL physical streaming replication ships write-ahead log (WAL) records directly to standby nodes.
Standby nodes replay WAL records to maintain sub-millisecond replication lag.
In the event of primary failure, standby promotion enables zero data loss failover.
"""

    ingest_res = await ingest_service.ingest_file(
        file_bytes=doc_text,
        filename="ha_guide.md",
    )
    assert ingest_res.status == "PARSED"

    # 2. Chunk document
    chunker = RecursiveTokenChunker(chunk_size=45, chunk_overlap=5)
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

    # 3. Vector indexing
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

    # 4. Set up hybrid retriever
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
        final_top_k=5,
    )

    # 5. Assemble RerankingPipeline with MockReranker
    reranker = MockReranker(model_name="mock-cross-encoder-v1")
    pipeline = RerankingPipeline(
        retriever=hybrid_retriever,
        reranker=reranker,
        candidates_k=5,
        top_k=2,
    )

    query = "PostgreSQL physical streaming replication WAL standby"
    results = await pipeline.search(query=query)

    assert len(results) == 2
    # Verify top result is Section C
    top_result = results[0]
    assert "Section C" in top_result.text or "replication" in top_result.text.lower()
    assert top_result.rerank_score is not None
    assert top_result.score == top_result.rerank_score
    # Verify dense and sparse scores preserved from earlier stages
    assert top_result.dense_score is not None
    assert top_result.sparse_score is not None
    assert top_result.metadata["reranker"] == "mock-cross-encoder-v1"

    # 6. Verify Graceful Fallback in Integration:
    failing_reranker = MockReranker(should_fail=True)
    fallback_pipeline = RerankingPipeline(
        retriever=hybrid_retriever,
        reranker=failing_reranker,
        candidates_k=5,
        top_k=2,
    )

    fallback_results = await fallback_pipeline.search(query=query)
    assert len(fallback_results) == 2
    # Should not have crashed; should return hybrid candidates with fallback metadata
    assert fallback_results[0].metadata["rerank_fallback"] is True
    assert "Mock reranking failure simulated" in fallback_results[0].metadata["rerank_error"]
    assert fallback_results[0].rerank_score is None
