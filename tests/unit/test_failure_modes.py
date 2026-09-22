"""Unit tests explicitly validating all 13 failure mode scenarios.

Scenarios tested:
1. invalid document
2. duplicate document
3. empty document
4. parser failure
5. embedding provider failure
6. vector database failure
7. reranker failure
8. LLM timeout
9. insufficient context
10. hallucinated answer
11. invalid citation
12. malformed request
13. unauthorized document access
"""

import uuid
from unittest.mock import AsyncMock

import pytest

from app.core.exceptions import (
    DatabaseError,
    EmbeddingError,
    EmbeddingTimeoutError,
    FileValidationError,
    ForbiddenError,
    GenerationTimeoutError,
    ParsingError,
    QueryValidationError,
    UnsupportedFileTypeError,
)
from app.models import Document, DocumentVersion
from app.rag.context.builder import ContextBuilder
from app.rag.context.domain import AssembledContext, ContextDocument
from app.rag.generation.domain import GeneratedResponse
from app.rag.generation.service import GenerationService
from app.rag.ingestion.parsers.docx import DocxParser
from app.rag.ingestion.parsers.pdf import PDFParser
from app.rag.ingestion.validator import FileValidator
from app.rag.reranking.pipeline import RerankingPipeline
from app.rag.retrieval.dense import DenseRetriever
from app.rag.validation.validator import GroundingValidator
from app.rag.vector.domain import RetrievalResult
from app.security.authorization import (
    DocumentAction,
    SecurityContext,
    TenantDocumentAccessControl,
)
from app.services.ingestion_service import IngestionService
from app.services.rag_service import RAGOrchestratorService
from tests.conftest import (
    FailingEmbeddingProvider,
    FailingLLMProvider,
    FailingReranker,
    FailingVectorStore,
)


# ============================================================================
# Scenario 1: Invalid Document
# ============================================================================
def test_scenario_1_invalid_document_unsupported_extension() -> None:
    """Validate that unsupported file extensions raise UnsupportedFileTypeError."""
    validator = FileValidator()
    with pytest.raises(UnsupportedFileTypeError) as exc:
        validator.validate(b"binary payload", "malicious_payload.exe")
    assert exc.value.status_code == 415
    assert ".exe" in str(exc.value)


def test_scenario_1_invalid_document_spoofed_pdf_magic_bytes() -> None:
    """Validate that files disguised as PDF without %PDF- signature raise FileValidationError."""
    validator = FileValidator()
    with pytest.raises(FileValidationError) as exc:
        validator.validate(b"NOT_A_PDF_STREAM_HEADER", "report.pdf")
    assert "Missing %PDF- file header signature" in str(exc.value)


# ============================================================================
# Scenario 2: Duplicate Document
# ============================================================================
@pytest.mark.asyncio
async def test_scenario_2_duplicate_document_skips_reprocessing() -> None:
    """Validate that IngestionService skips re-processing for duplicate content hashes."""
    doc_repo = AsyncMock()
    ver_repo = AsyncMock()
    job_repo = AsyncMock()
    storage = AsyncMock()

    doc_id = uuid.uuid4()
    existing_ver = DocumentVersion(
        id=uuid.uuid4(),
        document_id=doc_id,
        version_number=1,
        content_hash="mocked_hash",
        size_bytes=100,
        status="COMPLETED",
        metadata_json={"title": "Existing Doc"},
    )
    parent_doc = Document(id=doc_id, name="existing.txt", document_type="txt", source="local://")

    ver_repo.get_by_content_hash.return_value = existing_ver
    doc_repo.get_by_id.return_value = parent_doc
    job_repo.list_by_version.return_value = []
    job_repo.get_jobs_by_document.return_value = []

    service = IngestionService(
        document_repo=doc_repo,
        version_repo=ver_repo,
        job_repo=job_repo,
        storage_service=storage,
    )

    result = await service.ingest_file(
        file_bytes=b"Sample document content for hash test.",
        filename="existing.txt",
    )

    assert result.is_duplicate is True
    assert result.document_id == doc_id
    storage.save.assert_not_called()


# ============================================================================
# Scenario 3: Empty Document
# ============================================================================
def test_scenario_3_empty_document_raises_validation_error() -> None:
    """Validate that 0-byte uploaded files raise FileValidationError."""
    validator = FileValidator()
    with pytest.raises(FileValidationError) as exc:
        validator.validate(b"", "empty_document.txt")
    assert "Uploaded file is empty (0 bytes)" in str(exc.value)


# ============================================================================
# Scenario 4: Parser Failure
# ============================================================================
@pytest.mark.asyncio
async def test_scenario_4_parser_failure_pdf_corrupted_stream() -> None:
    """Validate that corrupted PDF streams raise ParsingError."""
    parser = PDFParser()
    with pytest.raises(ParsingError) as exc:
        await parser.parse(b"%PDF-corrupted-stream-bytes", "damaged.pdf")
    assert "Corrupted or invalid PDF format" in str(exc.value)


@pytest.mark.asyncio
async def test_scenario_4_parser_failure_docx_corrupted_archive() -> None:
    """Validate that corrupted DOCX ZIP archives raise ParsingError."""
    parser = DocxParser()
    with pytest.raises(ParsingError) as exc:
        await parser.parse(b"PK\x03\x04corrupted-archive", "damaged.docx")
    assert "Failed to open or parse DOCX" in str(exc.value)


# ============================================================================
# Scenario 5: Embedding Provider Failure
# ============================================================================
@pytest.mark.asyncio
async def test_scenario_5_embedding_provider_error() -> None:
    """Validate that embedding provider exceptions are raised with appropriate details."""
    provider = FailingEmbeddingProvider(mode="error")
    with pytest.raises(EmbeddingError) as exc:
        await provider.embed_text("Query text to embed")
    assert exc.value.status_code == 502
    assert "Simulated upstream embedding provider failure" in str(exc.value)


@pytest.mark.asyncio
async def test_scenario_5_embedding_provider_timeout() -> None:
    """Validate that embedding provider timeouts raise EmbeddingTimeoutError."""
    provider = FailingEmbeddingProvider(mode="timeout")
    with pytest.raises(EmbeddingTimeoutError) as exc:
        await provider.embed_text("Query text to embed")
    assert exc.value.status_code == 504
    assert "timed out" in str(exc.value)


# ============================================================================
# Scenario 6: Vector Database Failure
# ============================================================================
@pytest.mark.asyncio
async def test_scenario_6_vector_database_failure() -> None:
    """Validate that vector store failures raise DatabaseError during search."""
    store = FailingVectorStore()
    with pytest.raises(DatabaseError) as exc:
        await store.search(query_vector=[0.1] * 384, top_k=5)
    assert exc.value.status_code == 500
    assert "Simulated vector database connection failure" in str(exc.value)


@pytest.mark.asyncio
async def test_scenario_6_dense_retriever_propagates_vector_failure() -> None:
    """Validate that DenseRetriever propagates database failures when vector search fails."""
    embed_mock = AsyncMock()
    embed_mock.embed_text.return_value = [0.1] * 384
    store = FailingVectorStore()

    retriever = DenseRetriever(embedding_provider=embed_mock, vector_store=store)
    with pytest.raises(DatabaseError):
        await retriever.retrieve("distributed systems", top_k=5)


# ============================================================================
# Scenario 7: Reranker Failure & Graceful Fallback
# ============================================================================
@pytest.mark.asyncio
async def test_scenario_7_reranker_failure_triggers_graceful_fallback() -> None:
    """Validate that reranker failures gracefully fall back to original candidate chunks."""
    failing_reranker = FailingReranker(mode="error")
    cid1, cid2 = uuid.uuid4(), uuid.uuid4()
    did = uuid.uuid4()
    candidates = [
        RetrievalResult(chunk_id=cid1, document_id=did, text="First candidate", score=0.9),
        RetrievalResult(chunk_id=cid2, document_id=did, text="Second candidate", score=0.8),
    ]

    retriever_mock = AsyncMock()
    retriever_mock.retrieve.return_value = candidates

    pipeline = RerankingPipeline(retriever=retriever_mock, reranker=failing_reranker)
    fallback_results = await pipeline.retrieve(
        query="test query",
        top_k=2,
    )

    # Graceful fallback returns candidates intact
    assert len(fallback_results) == 2
    assert fallback_results[0].chunk_id == cid1
    assert fallback_results[1].chunk_id == cid2


@pytest.mark.asyncio
async def test_scenario_7_orchestrator_rerank_failure_fallback_metadata() -> None:
    """Validate orchestrator captures rerank fallback metadata when reranker times out."""
    retriever_mock = AsyncMock()
    retriever_mock.retrieve.return_value = [
        RetrievalResult(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            text="PostgreSQL WAL replication.",
            score=0.95,
        )
    ]
    failing_reranker = FailingReranker(mode="timeout")
    gen_service = AsyncMock()
    gen_service.generate_answer.return_value = GeneratedResponse(
        answer="PostgreSQL replicates via WAL [1].",
        citations=[1],
        confidence_score=0.9,
        grounded=True,
        insufficient_context=False,
    )

    orchestrator = RAGOrchestratorService(
        retriever=retriever_mock,
        context_builder=ContextBuilder(),
        generation_service=gen_service,
        reranker=failing_reranker,
        grounding_validator=None,
    )

    response = await orchestrator.execute(query="How does replication work?")
    assert response.metadata.get("rerank_fallback") is True
    assert "timed out" in response.metadata.get("rerank_error", "")
    assert response.answer == "PostgreSQL replicates via WAL [1]."


# ============================================================================
# Scenario 8: LLM Timeout
# ============================================================================
@pytest.mark.asyncio
async def test_scenario_8_llm_timeout() -> None:
    """Validate that LLM generation timeout raises GenerationTimeoutError."""
    llm = FailingLLMProvider(mode="timeout")
    service = GenerationService(llm_provider=llm)

    context = AssembledContext(
        formatted_context="[1] Context",
        documents=[],
        citation_map={},
        total_tokens=10,
        total_chunks=1,
        truncated=False,
        dropped_chunks_count=0,
    )

    with pytest.raises(GenerationTimeoutError) as exc:
        await service.generate_answer(query="Explain Raft", context=context)
    assert exc.value.status_code == 504
    assert "timed out" in str(exc.value)


# ============================================================================
# Scenario 9: Insufficient Context
# ============================================================================
@pytest.mark.asyncio
async def test_scenario_9_insufficient_context_honest_refusal() -> None:
    """Validate that model's explicit declaration of insufficient context is recognized."""
    empty_retriever = AsyncMock()
    empty_retriever.retrieve.return_value = []

    gen_service = AsyncMock()
    gen_service.generate_answer.return_value = GeneratedResponse(
        answer="I do not have sufficient information in the provided context to answer this question.",
        citations=[],
        confidence_score=0.0,
        grounded=True,
        insufficient_context=True,
    )

    orchestrator = RAGOrchestratorService(
        retriever=empty_retriever,
        context_builder=ContextBuilder(),
        generation_service=gen_service,
        grounding_validator=GroundingValidator(),
    )

    response = await orchestrator.execute(query="What is quantum gravity?")
    assert response.insufficient_context is True
    assert response.citations == []
    assert len(response.referenced_documents) == 0


@pytest.mark.asyncio
async def test_scenario_9_empty_context_unsupported_claim_forces_zero_confidence() -> None:
    """Validate that an answer claiming facts when context is empty forces confidence to 0.0 and refusal fallback."""
    empty_retriever = AsyncMock()
    empty_retriever.retrieve.return_value = []

    gen_service = AsyncMock()
    gen_service.generate_answer.return_value = GeneratedResponse(
        answer="Quantum gravity unifies relativity and quantum mechanics completely.",
        citations=[],
        confidence_score=0.9,
        grounded=False,
        insufficient_context=False,
    )

    orchestrator = RAGOrchestratorService(
        retriever=empty_retriever,
        context_builder=ContextBuilder(),
        generation_service=gen_service,
        grounding_validator=GroundingValidator(),
    )

    response = await orchestrator.execute(query="What is quantum gravity?")
    assert response.insufficient_context is True
    assert response.confidence_score == 0.0
    assert "I cannot answer this question with sufficient confidence" in response.answer


# ============================================================================
# Scenario 10: Hallucinated Answer
# ============================================================================
def test_scenario_10_hallucinated_answer_detected_and_refused() -> None:
    """Validate that hallucinated claims are detected by GroundingValidator and replaced with refusal."""
    doc = ContextDocument(
        citation_id=1,
        citation_label="[1]",
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        source_id="raft.md",
        title="Raft Protocol",
        page_number=1,
        score=0.9,
        content="The Raft protocol uses elected leaders to manage replicated state machines.",
        token_count=15,
    )
    context = AssembledContext(
        formatted_context="[1] The Raft protocol uses elected leaders to manage replicated state machines.",
        documents=[doc],
        citation_map={1: doc},
        total_tokens=15,
        total_chunks=1,
        truncated=False,
        dropped_chunks_count=0,
    )

    validator = GroundingValidator()
    # Answer hallucinates completely unrelated claims
    hallucinated_answer = (
        "The Raft protocol uses elected leaders [1]. "
        "It also automatically navigates interplanetary spacecraft across warp zones [1]."
    )

    result = validator.validate(query="What is Raft?", answer=hallucinated_answer, context=context)
    assert result.grounded is False
    assert result.fallback_applied is True
    assert len(result.unsupported_claims) >= 1
    assert "I cannot answer this question with sufficient confidence" in result.final_answer


# ============================================================================
# Scenario 11: Invalid Citation
# ============================================================================
def test_scenario_11_invalid_citation_detected() -> None:
    """Validate that citations to non-existent context identifiers are flagged as errors."""
    doc = ContextDocument(
        citation_id=1,
        citation_label="[1]",
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        source_id="ha.md",
        title="High Availability",
        page_number=1,
        score=0.9,
        content="Primary databases replicate asynchronously.",
        token_count=10,
    )
    context = AssembledContext(
        formatted_context="[1] Primary databases replicate asynchronously.",
        documents=[doc],
        citation_map={1: doc},
        total_tokens=10,
        total_chunks=1,
        truncated=False,
        dropped_chunks_count=0,
    )

    validator = GroundingValidator()
    # Citation [99] does not exist in context
    answer_with_invalid_cite = "Primary databases replicate asynchronously [99]."

    result = validator.validate(
        query="How does HA work?", answer=answer_with_invalid_cite, context=context
    )
    assert len(result.citation_errors) >= 1
    assert any("99" in err for err in result.citation_errors)
    assert result.fallback_applied is True


# ============================================================================
# Scenario 12: Malformed Request
# ============================================================================
@pytest.mark.asyncio
async def test_scenario_12_malformed_query_empty_or_whitespace() -> None:
    """Validate that empty or whitespace-only queries raise QueryValidationError."""
    retriever = AsyncMock()
    gen_service = AsyncMock()

    orchestrator = RAGOrchestratorService(
        retriever=retriever,
        context_builder=ContextBuilder(),
        generation_service=gen_service,
    )

    with pytest.raises(QueryValidationError) as exc:
        await orchestrator.execute(query="    \n\t  ")
    assert exc.value.status_code == 400
    assert "Query must not be empty" in str(exc.value)


# ============================================================================
# Scenario 13: Unauthorized Document Access
# ============================================================================
def test_scenario_13_unauthorized_cross_tenant_document_access() -> None:
    """Validate that TenantDocumentAccessControl blocks cross-tenant document operations with ForbiddenError."""
    policy = TenantDocumentAccessControl(tenant_enforcement_enabled=True)
    caller_ctx = SecurityContext(user_id="alice", tenant_id="tenant_alpha", roles=("user",))
    target_doc_meta = {"tenant_id": "tenant_beta", "classification": "confidential"}

    with pytest.raises(ForbiddenError) as exc:
        policy.authorize_document(caller_ctx, target_doc_meta, DocumentAction.READ)
    assert exc.value.status_code == 403
    assert "Document belongs to tenant 'tenant_beta'" in str(exc.value)
