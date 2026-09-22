"""End-to-end integration tests for complete RAG pipeline: Ingestion to Answer Generation."""

from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.chunking.recursive import RecursiveTokenChunker
from app.rag.context.builder import ContextBuilder
from app.rag.embeddings.mock import MockEmbeddingProvider
from app.rag.embeddings.vector_interfaces import VectorRecord
from app.rag.generation.mock import MockLLMProvider
from app.rag.generation.service import GenerationService
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
async def test_end_to_end_rag_generation_pipeline(
    db_session: AsyncSession,
    document_repo: SQLAlchemyDocumentRepository,
    version_repo: SQLAlchemyDocumentVersionRepository,
    chunk_repo: SQLAlchemyDocumentChunkRepository,
    job_repo: SQLAlchemyIngestionJobRepository,
    tmp_path: Path,
) -> None:
    """Verify complete end-to-end RAG workflow: Ingest -> Chunk -> Index -> Hybrid -> Rerank -> Context -> LLM Answer."""
    # 1. Ingestion
    storage = LocalStorageService(base_dir=tmp_path)
    ingest_service = IngestionService(
        document_repo=document_repo,
        version_repo=version_repo,
        job_repo=job_repo,
        storage_service=storage,
    )

    doc_text = b"""# Database Internals Guide

## Section A: Write-Ahead Logging
PostgreSQL write-ahead logs (WAL) guarantee write atomicity and durability across sudden system reboots.

## Section B: Raft Consensus Engine
Raft maintains state machine replicas using heartbeats and synchronized log indices.
"""

    ingest_res = await ingest_service.ingest_file(
        file_bytes=doc_text,
        filename="db_internals.md",
    )
    assert ingest_res.status == "PARSED"

    # 2. Chunking
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

    # 3. Embedding and Vector Indexing
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
        embedding_provider=embedding_provider, vector_store=vector_store
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
    reranker = MockReranker(model_name="mock-reranker-v1")
    search_pipeline = RerankingPipeline(
        retriever=hybrid_retriever,
        reranker=reranker,
        candidates_k=5,
        top_k=3,
    )

    query = "PostgreSQL write-ahead logging durability"
    reranked = await search_pipeline.search(query=query)
    assert len(reranked) >= 1

    # 6. Context Assembly
    context_builder = ContextBuilder(max_tokens=1000, max_chunks=3)
    assembled = context_builder.build_context(query=query, candidates=reranked)
    assert assembled.total_chunks >= 1
    assert assembled.total_tokens <= 1000

    # 7. LLM Answer Generation
    llm_provider = MockLLMProvider(model_name="mock-gpt-4o")
    gen_service = GenerationService(llm_provider=llm_provider)

    response = await gen_service.generate_answer(query=query, context=assembled)

    # 8. Verifications
    assert response.grounded is True
    assert response.insufficient_context is False
    assert response.confidence_score > 0.0
    assert len(response.citations) >= 1
    assert "[1]" in response.answer
    assert len(response.referenced_documents) == len(response.citations)
    assert response.referenced_documents[0].citation_id == 1
    assert response.referenced_documents[0].document_id == ingest_res.document_id

    # 9. Verify Explicit Uncertainty on Insufficient Context
    empty_context = context_builder.build_context(query="unrelated query", candidates=[])
    uncertain_response = await gen_service.generate_answer(
        query="Quantum computing entanglement", context=empty_context
    )

    assert uncertain_response.insufficient_context is True
    assert uncertain_response.confidence_score == 0.0
    assert uncertain_response.citations == []
    assert "I do not have sufficient information" in uncertain_response.answer
