"""Integration tests for end-to-end Context Assembly with database, hybrid search, and reranking."""

from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.chunking.recursive import RecursiveTokenChunker
from app.rag.context.builder import ContextBuilder
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
async def test_end_to_end_context_assembly_pipeline(
    db_session: AsyncSession,
    document_repo: SQLAlchemyDocumentRepository,
    version_repo: SQLAlchemyDocumentVersionRepository,
    chunk_repo: SQLAlchemyDocumentChunkRepository,
    job_repo: SQLAlchemyIngestionJobRepository,
    tmp_path: Path,
) -> None:
    """Verify complete pipeline: Ingest -> Chunk -> Index -> Hybrid -> Rerank -> Context Assembly -> Citation Resolution."""
    # 1. Ingest multi-section technical document
    storage = LocalStorageService(base_dir=tmp_path)
    ingest_service = IngestionService(
        document_repo=document_repo,
        version_repo=version_repo,
        job_repo=job_repo,
        storage_service=storage,
    )

    doc_text = b"""# Storage & Distributed Systems Guide

## Section A: Multi-Version Concurrency
PostgreSQL isolates concurrent transactions through snapshot isolation and write-ahead logs.
Visibility maps track dirty tuple visibility across parallel query executions.

## Section B: Raft Distributed Consensus
Raft elects a cluster leader to manage consistent replicated state machine transitions.
Followers accept log entries appended by the active leader.

## Section C: Redis Caching and Eviction
Redis stores keys in memory with configurable LRU and LFU eviction policies.
"""

    ingest_res = await ingest_service.ingest_file(
        file_bytes=doc_text,
        filename="systems_guide.md",
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

    # 4. Hybrid Retriever
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

    # 5. Reranking Pipeline
    reranker = MockReranker(model_name="test-reranker-v1")
    search_pipeline = RerankingPipeline(
        retriever=hybrid_retriever,
        reranker=reranker,
        candidates_k=5,
        top_k=3,
    )

    query = "PostgreSQL multi-version concurrency snapshot isolation"
    reranked_candidates = await search_pipeline.search(query=query)
    assert len(reranked_candidates) >= 1

    # 6. Context Assembly Stage
    builder = ContextBuilder(max_tokens=800, max_chunks=3)
    assembled_context = builder.build_context(query=query, candidates=reranked_candidates)

    assert assembled_context.total_chunks > 0
    assert assembled_context.total_tokens <= 800
    assert len(assembled_context.documents) == assembled_context.total_chunks

    # Top document should be Section A
    top_doc = assembled_context.documents[0]
    assert top_doc.citation_id == 1
    assert top_doc.citation_label == "[1]"
    assert "concurrency" in top_doc.content.lower() or "snapshot" in top_doc.content.lower()
    assert top_doc.score > 0.0
    assert top_doc.source_id != ""

    # Verify citation manifest
    manifest = assembled_context.format_citation_manifest()
    assert '[1] "' in manifest
    assert "Score:" in manifest

    # 7. Simulate Generation layer referencing citations
    simulated_llm_response = (
        "PostgreSQL uses snapshot isolation for concurrent transactions [1]. "
        "Non-existent citation [42] should be ignored."
    )

    resolved_sources = assembled_context.resolve_citations(simulated_llm_response)
    assert len(resolved_sources) == 1
    assert resolved_sources[0].citation_id == 1
    assert resolved_sources[0].chunk_id == top_doc.chunk_id
    assert resolved_sources[0].document_id == ingest_res.document_id
