"""Integration tests for full RAG pipeline with Grounding Validator guardrails."""

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


@pytest.mark.asyncio
async def test_full_rag_pipeline_with_grounding_guardrail(
    db_session: AsyncSession,
    document_repo: SQLAlchemyDocumentRepository,
    version_repo: SQLAlchemyDocumentVersionRepository,
    chunk_repo: SQLAlchemyDocumentChunkRepository,
    job_repo: SQLAlchemyIngestionJobRepository,
    tmp_path: Path,
) -> None:
    """Verify entire workflow: Ingestion -> Chunking -> Indexing -> Hybrid -> Rerank -> Context -> LLM -> Grounding Guardrail."""
    # 1. Ingest document
    storage = LocalStorageService(base_dir=tmp_path)
    ingest_service = IngestionService(
        document_repo=document_repo,
        version_repo=version_repo,
        job_repo=job_repo,
        storage_service=storage,
    )

    doc_bytes = b"""# Distributed Consensus Protocol Specifications

## Section A: Raft Replicated Logs
The Raft consensus algorithm enforces replicated log consistency across distributed server clusters.
Leaders append log entries and replicate them to follower state machines before broadcasting commit index updates.
"""

    ingest_res = await ingest_service.ingest_file(
        file_bytes=doc_bytes,
        filename="consensus_spec.md",
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
    assert len(chunks) >= 1

    # 3. Embed & Index
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

    # 4. Hybrid Retrieval & Reranking
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
    search_pipeline = RerankingPipeline(
        retriever=hybrid_retriever,
        reranker=reranker,
        candidates_k=5,
        top_k=2,
    )

    query = "Raft replicated logs consensus"
    candidates = await search_pipeline.search(query=query)
    assert len(candidates) >= 1

    # 5. Context Assembly
    context_builder = ContextBuilder(max_tokens=1000, max_chunks=2)
    assembled = context_builder.build_context(query=query, candidates=candidates)

    # 6. LLM Generation
    llm_provider = MockLLMProvider()
    gen_service = GenerationService(llm_provider=llm_provider)
    generated_resp = await gen_service.generate_answer(query=query, context=assembled)

    # 7. Grounding Guardrail Validation on Generated Answer
    validator = GroundingValidator()
    val_result = validator.validate(
        query=query,
        answer=generated_resp.answer,
        context=assembled,
    )

    assert val_result.grounded is True
    assert val_result.confidence_score >= 0.70
    assert val_result.fallback_applied is False
    assert val_result.final_answer == generated_resp.answer

    # 8. Guardrail Interception of Injected Hallucination
    hallucinated_answer = (
        "Raft consensus requires nuclear cold fusion reactors to power clusters [1]."
    )
    hallucinated_val = validator.validate(
        query=query,
        answer=hallucinated_answer,
        context=assembled,
    )

    assert hallucinated_val.grounded is False
    assert hallucinated_val.fallback_applied is True
    assert (
        "I cannot answer this question with sufficient confidence" in hallucinated_val.final_answer
    )
    assert len(hallucinated_val.unsupported_claims) >= 1
