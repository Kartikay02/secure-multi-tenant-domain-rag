"""Integration tests for the complete end-to-end RAG Orchestrator pipeline."""

from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.chunking.recursive import RecursiveTokenChunker
from app.rag.context.builder import ContextBuilder
from app.rag.embeddings.mock import MockEmbeddingProvider
from app.rag.embeddings.vector_interfaces import VectorRecord
from app.rag.generation.mock import MockLLMProvider
from app.rag.generation.service import GenerationService
from app.rag.query.preprocessor import StandardQueryPreprocessor
from app.rag.reranking.mock import MockReranker
from app.rag.retrieval.dense import DenseRetriever
from app.rag.retrieval.fusion import ReciprocalRankFusion
from app.rag.retrieval.hybrid import HybridRetriever
from app.rag.retrieval.lexical import PGLexicalRetriever
from app.rag.storage.local import LocalStorageService
from app.rag.validation.validator import GroundingValidator
from app.rag.vector.pgvector import PGVectorStore
from app.repositories import (
    SQLAlchemyDocumentChunkRepository,
    SQLAlchemyDocumentRepository,
    SQLAlchemyDocumentVersionRepository,
    SQLAlchemyIngestionJobRepository,
)
from app.services.chunking_service import ChunkingService
from app.services.ingestion_service import IngestionService
from app.services.rag_service import RAGOrchestratorService, create_rag_orchestrator


@pytest.fixture
async def seeded_rag_environment(
    db_session: AsyncSession,
    document_repo: SQLAlchemyDocumentRepository,
    version_repo: SQLAlchemyDocumentVersionRepository,
    chunk_repo: SQLAlchemyDocumentChunkRepository,
    job_repo: SQLAlchemyIngestionJobRepository,
    tmp_path: Path,
) -> tuple[HybridRetriever, MockReranker, ContextBuilder, GenerationService, GroundingValidator]:
    """Ingest, chunk, embed, and index domain documents, returning ready-to-wire pipeline stages."""
    storage = LocalStorageService(base_dir=tmp_path)
    ingest_service = IngestionService(
        document_repo=document_repo,
        version_repo=version_repo,
        job_repo=job_repo,
        storage_service=storage,
    )

    doc_content = b"""# High-Performance Retrieval Architectures

## Dense Vector Indexing with HNSW
Hierarchical Navigable Small World (HNSW) graphs organize high-dimensional vectors into multi-layer proximity networks.
HNSW achieves sub-millisecond approximate nearest neighbor query latency by pruning search paths exponentially.

## Lexical Inverted Indexes
BM25 and full-text inverted indexes score exact lexical token matches using term frequency and inverse document frequency.
Hybrid fusion combines dense and lexical retrieval to achieve maximum domain recall.
"""

    ingest_res = await ingest_service.ingest_file(
        file_bytes=doc_content,
        filename="retrieval_architecture.md",
    )
    assert ingest_res.status == "PARSED"

    # Chunking
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

    # Embedding & PGVector Indexing
    embedding_provider = MockEmbeddingProvider(dimension=128, model_name="bge-mock-dim128")
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

    # Retrieval Components
    dense_retriever = DenseRetriever(
        embedding_provider=embedding_provider, vector_store=vector_store
    )
    lexical_retriever = PGLexicalRetriever(session=db_session)
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        lexical_retriever=lexical_retriever,
        fusion_strategy=ReciprocalRankFusion(k=60),
    )

    reranker = MockReranker(model_name="mock-reranker-v1")
    context_builder = ContextBuilder(max_tokens=1500, max_chunks=5)
    llm_provider = MockLLMProvider()
    gen_service = GenerationService(llm_provider=llm_provider)
    validator = GroundingValidator()

    return hybrid_retriever, reranker, context_builder, gen_service, validator


@pytest.mark.asyncio
async def test_e2e_rag_orchestrator_query_execution(
    seeded_rag_environment: tuple[
        HybridRetriever, MockReranker, ContextBuilder, GenerationService, GroundingValidator
    ],
) -> None:
    """Verify complete end-to-end pipeline execution from raw user query to grounded RAGResponse."""
    hybrid_retriever, reranker, context_builder, gen_service, validator = seeded_rag_environment

    preprocessor = StandardQueryPreprocessor(min_length=2, max_length=1000)
    orchestrator = RAGOrchestratorService(
        retriever=hybrid_retriever,
        reranker=reranker,
        context_builder=context_builder,
        generation_service=gen_service,
        grounding_validator=validator,
        query_preprocessor=preprocessor,
        default_retrieval_top_k=5,
        default_rerank_top_k=2,
    )

    raw_query = "   How does HNSW achieve sub-millisecond query latency?   \n"
    response = await orchestrator.execute(
        query=raw_query,
        request_id="e2e-trace-001",
    )

    # 1. Verification of query preprocessing
    assert response.query == "How does HNSW achieve sub-millisecond query latency?"
    assert response.raw_query == raw_query
    assert response.request_id == "e2e-trace-001"

    # 2. Verification of retrieved and assembled content
    assert response.retrieved_chunks_count > 0
    assert response.context_chunks_count > 0
    assert response.context_tokens > 0

    # 3. Verification of generation & grounding validation
    assert response.answer != ""
    assert response.grounded is True
    assert response.confidence_score >= 0.70
    assert response.fallback_applied is False
    assert len(response.citations) >= 1

    # 4. Verification of citation resolution and manifest
    assert len(response.referenced_documents) >= 1
    assert "References:" in response.citation_manifest
    assert response.referenced_documents[0].title != ""

    # 5. Verification of stage latencies
    latencies = response.stage_latencies
    assert latencies.preprocessing_ms >= 0.0
    assert latencies.retrieval_ms > 0.0
    assert latencies.reranking_ms >= 0.0
    assert latencies.context_assembly_ms >= 0.0
    assert latencies.generation_ms >= 0.0
    assert latencies.validation_ms >= 0.0
    assert latencies.total_ms > 0.0

    # Telemetry metadata
    assert response.metadata.get("reranker_applied") is True


@pytest.mark.asyncio
async def test_e2e_rag_orchestrator_with_reranking_disabled(
    seeded_rag_environment: tuple[
        HybridRetriever, MockReranker, ContextBuilder, GenerationService, GroundingValidator
    ],
) -> None:
    """Verify end-to-end execution when reranking is disabled at runtime."""
    hybrid_retriever, reranker, context_builder, gen_service, validator = seeded_rag_environment

    orchestrator = RAGOrchestratorService(
        retriever=hybrid_retriever,
        reranker=reranker,
        context_builder=context_builder,
        generation_service=gen_service,
        grounding_validator=validator,
    )

    response = await orchestrator.execute(
        query="Explain lexical inverted indexes and BM25",
        enable_reranking=False,
    )

    assert response.grounded is True
    assert response.metadata.get("reranker_applied") is False
    assert response.answer != ""
    assert len(response.citations) >= 1


@pytest.mark.asyncio
async def test_e2e_create_rag_orchestrator_factory(
    seeded_rag_environment: tuple[
        HybridRetriever, MockReranker, ContextBuilder, GenerationService, GroundingValidator
    ],
) -> None:
    """Verify create_rag_orchestrator factory constructs an operational pipeline."""
    hybrid_retriever, reranker, context_builder, gen_service, validator = seeded_rag_environment

    orchestrator = create_rag_orchestrator(
        retriever=hybrid_retriever,
        reranker=reranker,
        context_builder=context_builder,
        generation_service=gen_service,
        grounding_validator=validator,
    )

    response = await orchestrator.execute(
        query="What is hybrid fusion in retrieval architectures?",
    )

    assert response.grounded is True
    assert response.answer != ""
    assert response.stage_latencies.total_ms > 0.0


@pytest.mark.asyncio
async def test_e2e_rag_orchestrator_insufficient_context_empty_results(
    seeded_rag_environment: tuple[
        HybridRetriever, MockReranker, ContextBuilder, GenerationService, GroundingValidator
    ],
) -> None:
    """Verify pipeline gracefully handles zero candidate retrieval (out-of-domain filter)."""
    hybrid_retriever, reranker, context_builder, gen_service, validator = seeded_rag_environment

    orchestrator = RAGOrchestratorService(
        retriever=hybrid_retriever,
        reranker=reranker,
        context_builder=context_builder,
        generation_service=gen_service,
        grounding_validator=validator,
    )

    # Apply filter that matches zero documents
    response = await orchestrator.execute(
        query="What is quantum entanglement cryptography?",
        filter_metadata={"section": "nonexistent_section_xyz"},
    )

    assert response.retrieved_chunks_count == 0
    assert response.context_chunks_count == 0
    assert response.insufficient_context is True
    assert response.citations == []
    assert response.referenced_documents == []
    assert "not have sufficient information" in response.answer.lower()
