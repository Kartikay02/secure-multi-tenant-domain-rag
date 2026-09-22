"""Unit regression tests for structured document answer synthesis (OpenAPI & Generic Nested JSON).

Covers all 10 required test cases:
TEST 1  - Natural-language synthesis for OpenAPI (not a raw dump).
TEST 2  - Answer contains valid citations pointing to retrieved evidence.
TEST 3  - Grounding validation passes for synthesized structured answers.
TEST 4  - Document-scoped queries retrieve and answer ONLY from the selected document.
TEST 5  - Missing/irrelevant structured questions safely refuse rather than hallucinate.
TEST 6  - Explicit request for raw content preserves raw format when requested.
TEST 7  - Chunks with multiple endpoints do not produce duplicated sentences or repeated lists.
TEST 8  - Tenant isolation remains enforced for structured documents.
TEST 9  - Mandatory grounding validation cannot be disabled or bypassed.
TEST 10 - Generic nested JSON synthesis proves the solution generalizes.
"""

import json
import uuid
from typing import Any

import pytest

from app.rag.context.builder import ContextBuilder
from app.rag.generation.mock import MockLLMProvider
from app.rag.generation.service import GenerationService
from app.rag.ingestion.parsers.json_parser import JSONParser
from app.rag.retrieval.interfaces import RetrieverProtocol
from app.rag.validation.validator import GroundingValidator
from app.rag.vector.domain import RetrievalResult
from app.services.rag_service import RAGOrchestratorService

OPENAPI_SAMPLE = {
    "openapi": "3.1.0",
    "info": {
        "title": "Domain RAG Service API",
        "version": "1.0.0",
        "description": "Production retrieval-augmented generation API service.",
    },
    "paths": {
        "/health": {
            "get": {
                "summary": "Health status probe",
                "description": "Service health check probe returning subsystem operational status.",
                "tags": ["System Health"],
                "responses": {"200": {"description": "System is healthy"}},
            }
        },
        "/api/v1/query": {
            "post": {
                "summary": "Execute grounded RAG query",
                "description": "Retrieves semantic contexts and generates grounded answers.",
                "tags": ["Query Operations"],
                "responses": {"200": {"description": "Grounded answer with citations"}},
            }
        },
        "/metrics": {
            "get": {
                "summary": "Prometheus telemetry metrics",
                "description": "Exposes runtime Prometheus performance metrics.",
                "tags": ["Observability"],
                "responses": {"200": {"description": "Prometheus text exposition"}},
            }
        },
    },
}

GENERIC_NESTED_JSON_SAMPLE = {
    "cluster": {
        "name": "production-us-east",
        "engine": "Kubernetes",
        "node_count": 16,
        "autoscaling": {
            "enabled": True,
            "min_nodes": 4,
            "max_nodes": 64,
        },
    },
    "database": {
        "engine": "PostgreSQL",
        "version": "16.2",
        "replication_mode": "physical streaming",
    },
}


class InMemoryTestRetriever(RetrieverProtocol):
    """In-memory candidate retriever for testing scoping and tenant filtering."""

    def __init__(self, candidates: list[RetrievalResult]) -> None:
        self.candidates = candidates

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        results = list(self.candidates)
        if filter_metadata:
            if "document_id" in filter_metadata:
                expected_doc = str(filter_metadata["document_id"])
                results = [r for r in results if str(r.document_id) == expected_doc]
            if "tenant_id" in filter_metadata:
                expected_tenant = str(filter_metadata["tenant_id"])
                results = [
                    r
                    for r in results
                    if r.metadata and str(r.metadata.get("tenant_id")) == expected_tenant
                ]
        if score_threshold is not None:
            results = [r for r in results if r.score >= score_threshold]
        return results[:top_k]


@pytest.mark.asyncio
async def test_openapi_natural_language_answer_not_raw_dump() -> None:
    """TEST 1: Given an OpenAPI document, answer is natural-language synthesis, not a raw dump."""
    parser = JSONParser()
    parsed = await parser.parse(json.dumps(OPENAPI_SAMPLE).encode(), "openapi.json")

    # Build context from parsed sections
    doc_id = uuid.uuid4()
    chunks = [
        RetrievalResult(
            chunk_id=uuid.uuid4(),
            document_id=doc_id,
            text=sec["content"],
            score=0.95,
            metadata={"title": sec["title"], "source": "openapi.json"},
        )
        for sec in parsed.sections
    ]

    builder = ContextBuilder()
    context = builder.build_context(query="What endpoints are available in the API?", candidates=chunks)

    provider = MockLLMProvider()
    service = GenerationService(llm_provider=provider)

    response = await service.generate_answer(
        query="What endpoints are available in the API?",
        context=context,
    )

    # Must NOT be a raw dump of keypaths or raw json
    assert "paths/health.get.tags:" not in response.answer
    assert "openapi: 3.1.0" not in response.answer or "OpenAPI 3.1.0 [" in response.answer
    assert "paths." not in response.answer

    # Must be structured natural language
    assert "The API exposes several endpoints" in response.answer or "- GET /health" in response.answer
    assert "/health" in response.answer
    assert "/api/v1/query" in response.answer
    assert len(response.citations) >= 1


@pytest.mark.asyncio
async def test_openapi_answer_valid_citations() -> None:
    """TEST 2: Answer contains valid citations pointing to retrieved evidence chunks."""
    parser = JSONParser()
    parsed = await parser.parse(json.dumps(OPENAPI_SAMPLE).encode(), "openapi.json")

    doc_id = uuid.uuid4()
    chunks = [
        RetrievalResult(
            chunk_id=uuid.uuid4(),
            document_id=doc_id,
            text=sec["content"],
            score=0.90,
            metadata={"title": sec["title"], "source": "openapi.json"},
        )
        for sec in parsed.sections
    ]

    builder = ContextBuilder()
    context = builder.build_context(query="List endpoints", candidates=chunks)

    service = GenerationService(llm_provider=MockLLMProvider())
    response = await service.generate_answer(query="List endpoints", context=context)

    # Check citations exist and resolve correctly
    assert len(response.citations) > 0
    for cid in response.citations:
        assert cid in context.citation_map
        doc = context.citation_map[cid]
        assert doc.source_id == "openapi.json"

    # Referenced documents list matches citations
    assert len(response.referenced_documents) == len(response.citations)
    assert {d.citation_id for d in response.referenced_documents} == set(response.citations)


@pytest.mark.asyncio
async def test_openapi_answer_remains_grounded() -> None:
    """TEST 3: Grounding validation passes for synthesized structured answers."""
    parser = JSONParser()
    parsed = await parser.parse(json.dumps(OPENAPI_SAMPLE).encode(), "openapi.json")

    chunks = [
        RetrievalResult(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            text=sec["content"],
            score=0.95,
            metadata={"title": sec["title"], "source": "openapi.json"},
        )
        for sec in parsed.sections
    ]

    context = ContextBuilder().build_context(query="What endpoints are available?", candidates=chunks)
    service = GenerationService(llm_provider=MockLLMProvider())
    response = await service.generate_answer(query="What endpoints are available?", context=context)

    # Run actual GroundingValidator
    validator = GroundingValidator()
    val_res = validator.validate(
        query="What endpoints are available?",
        answer=response.answer,
        context=context,
    )

    assert val_res.grounded is True
    assert val_res.confidence_score >= 0.70
    assert len(val_res.unsupported_claims) == 0
    assert len(val_res.citation_errors) == 0


@pytest.mark.asyncio
async def test_document_scoped_query_isolation() -> None:
    """TEST 4: Document-scoped queries retrieve and answer ONLY from the selected document."""
    doc_a_id = uuid.uuid4()
    doc_b_id = uuid.uuid4()

    chunk_a = RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=doc_a_id,
        text="### Endpoint: GET /health\n- Summary: System health check probe",
        score=0.95,
        metadata={"title": "GET /health", "source": "openapi_a.json", "document_id": str(doc_a_id)},
    )
    chunk_b = RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=doc_b_id,
        text="### Endpoint: POST /billing/charge\n- Summary: Process credit card payments",
        score=0.90,
        metadata={"title": "POST /billing/charge", "source": "billing.json", "document_id": str(doc_b_id)},
    )

    retriever = InMemoryTestRetriever([chunk_a, chunk_b])
    builder = ContextBuilder()
    gen_service = GenerationService(llm_provider=MockLLMProvider())
    validator = GroundingValidator()

    orchestrator = RAGOrchestratorService(
        retriever=retriever,
        context_builder=builder,
        generation_service=gen_service,
        grounding_validator=validator,
    )

    # Query scoped strictly to doc_a
    response = await orchestrator.execute(
        query="What endpoints are available?",
        filter_metadata={"document_id": str(doc_a_id)},
    )

    # Must only contain doc_a evidence
    assert "/health" in response.answer
    assert "billing" not in response.answer.lower()
    for ref_doc in response.referenced_documents:
        assert ref_doc.document_id == doc_a_id
        assert ref_doc.document_id != doc_b_id


@pytest.mark.asyncio
async def test_insufficient_evidence_safe_refusal() -> None:
    """TEST 5: Missing/irrelevant structured questions safely refuse rather than hallucinate."""
    # Context only contains health and metrics
    chunk = RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        text="### Endpoint: GET /health\n- Summary: Health probe",
        score=0.95,
        metadata={"title": "GET /health", "source": "openapi.json"},
    )
    context = ContextBuilder().build_context(query="banking api secret", candidates=[chunk])

    gen_service = GenerationService(llm_provider=MockLLMProvider())

    response = await gen_service.generate_answer(
        query="What is the banking api secret endpoint?",
        context=context,
    )

    # Must safely refuse rather than hallucinating banking secrets
    assert response.insufficient_context is True or "do not have sufficient information" in response.answer.lower()


@pytest.mark.asyncio
async def test_explicit_raw_content_request_allowed() -> None:
    """TEST 6: Explicit request for raw content preserves raw format when requested."""
    raw_openapi_content = '{"openapi": "3.1.0", "info": {"title": "Raw API"}}'
    chunk = RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        text=raw_openapi_content,
        score=0.95,
        metadata={"title": "Raw Schema", "source": "openapi.json"},
    )
    context = ContextBuilder().build_context(query="Show me the raw OpenAPI definition", candidates=[chunk])

    gen_service = GenerationService(llm_provider=MockLLMProvider())
    response = await gen_service.generate_answer(
        query="Show me the raw OpenAPI definition",
        context=context,
    )

    # Must preserve code block
    assert "```" in response.answer
    assert '{"openapi": "3.1.0"' in response.answer
    assert len(response.citations) >= 1


@pytest.mark.asyncio
async def test_multiple_chunks_no_duplicate_answer_content() -> None:
    """TEST 7: Chunks with multiple or overlapping endpoints do not duplicate list items."""
    doc_id = uuid.uuid4()
    # Duplicate endpoint information across overlapping chunks
    chunk_1 = RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        text="### Endpoint: GET /health\n- Summary: System health check probe\n\n### Endpoint: POST /query\n- Summary: RAG query execution",
        score=0.95,
        metadata={"title": "Endpoints 1", "source": "openapi.json"},
    )
    chunk_2 = RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        text="### Endpoint: GET /health\n- Summary: System health check probe\n\n### Endpoint: GET /metrics\n- Summary: Telemetry metrics",
        score=0.90,
        metadata={"title": "Endpoints 2", "source": "openapi.json"},
    )

    context = ContextBuilder().build_context(query="What endpoints are available?", candidates=[chunk_1, chunk_2])
    gen_service = GenerationService(llm_provider=MockLLMProvider())
    response = await gen_service.generate_answer(query="What endpoints are available?", context=context)

    # The GET /health endpoint should only appear once in bullet list
    lines = [line.strip() for line in response.answer.split("\n") if line.strip().startswith("- ")]
    health_bullets = [line for line in lines if "/health" in line]
    assert len(health_bullets) == 1, f"Expected 1 /health bullet, got {len(health_bullets)}: {lines}"


@pytest.mark.asyncio
async def test_tenant_isolation_intact() -> None:
    """TEST 8: Tenant isolation remains enforced for structured documents."""
    doc_tenant_a = uuid.uuid4()
    doc_tenant_b = uuid.uuid4()

    chunk_a = RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=doc_tenant_a,
        text="### Endpoint: GET /tenant_a/secrets\n- Summary: Tenant A secrets",
        score=0.95,
        metadata={"tenant_id": "tenant_a", "title": "Tenant A API", "source": "api_a.json"},
    )
    chunk_b = RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=doc_tenant_b,
        text="### Endpoint: GET /tenant_b/secrets\n- Summary: Tenant B secrets",
        score=0.98,  # Higher score, but wrong tenant!
        metadata={"tenant_id": "tenant_b", "title": "Tenant B API", "source": "api_b.json"},
    )

    # Retriever returns both candidates
    retriever = InMemoryTestRetriever([chunk_a, chunk_b])
    orchestrator = RAGOrchestratorService(
        retriever=retriever,
        context_builder=ContextBuilder(),
        generation_service=GenerationService(llm_provider=MockLLMProvider()),
        grounding_validator=GroundingValidator(),
        tenant_enforcement_enabled=True,
    )

    # Query as tenant_a
    response = await orchestrator.execute(
        query="What endpoints are available?",
        tenant_id="tenant_a",
    )

    # tenant_b content must NEVER be present
    assert "tenant_b" not in response.answer
    assert all(d.document_id == doc_tenant_a for d in response.referenced_documents)


@pytest.mark.asyncio
async def test_mandatory_grounding_cannot_be_disabled() -> None:
    """TEST 9: Mandatory grounding validation cannot be disabled or bypassed by client."""
    chunk = RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        text="### Endpoint: GET /health\n- Summary: Health check",
        score=0.95,
        metadata={"title": "Health API", "source": "openapi.json"},
    )

    retriever = InMemoryTestRetriever([chunk])
    validator = GroundingValidator()
    orchestrator = RAGOrchestratorService(
        retriever=retriever,
        context_builder=ContextBuilder(),
        generation_service=GenerationService(llm_provider=MockLLMProvider()),
        grounding_validator=validator,
        mandatory_grounding=True,
    )

    # Client explicitly attempts to disable grounding
    response = await orchestrator.execute(
        query="What endpoints are available?",
        enable_grounding=False,
    )

    # Server enforced mandatory grounding
    assert response.grounded is True
    assert response.confidence_score > 0.0


@pytest.mark.asyncio
async def test_generic_nested_json_synthesis() -> None:
    """TEST 10: Generic nested JSON document synthesizes natural language rather than raw dump."""
    parser = JSONParser()
    raw_json_bytes = json.dumps(GENERIC_NESTED_JSON_SAMPLE).encode()
    parsed = await parser.parse(raw_json_bytes, "infrastructure_config.json")

    # Ingest into context
    doc_id = uuid.uuid4()
    chunks = [
        RetrievalResult(
            chunk_id=uuid.uuid4(),
            document_id=doc_id,
            text=sec["content"],
            score=0.92,
            metadata={"title": sec["title"], "source": "infrastructure_config.json"},
        )
        for sec in parsed.sections
    ]

    context = ContextBuilder().build_context(
        query="What is the cluster configuration and autoscaling settings?",
        candidates=chunks,
    )

    service = GenerationService(llm_provider=MockLLMProvider())
    response = await service.generate_answer(
        query="What is the cluster configuration and autoscaling settings?",
        context=context,
    )

    # Must be natural language synthesis
    assert len(response.answer) > 20
    assert "{" not in response.answer  # Not raw JSON syntax
    assert "}" not in response.answer
    assert len(response.citations) >= 1

    # Must pass grounding validation
    val_res = GroundingValidator().validate(
        query="What is the cluster configuration and autoscaling settings?",
        answer=response.answer,
        context=context,
    )
    assert val_res.grounded is True
    assert val_res.confidence_score >= 0.70


@pytest.mark.asyncio
async def test_answer_contains_no_flattened_internal_keypaths() -> None:
    """TEST 11: Assert that answers for structured documents contain no flattened internal keypaths."""
    parser = JSONParser()
    raw_json_bytes = json.dumps(OPENAPI_SAMPLE).encode()
    parsed = await parser.parse(raw_json_bytes, "openapi.json")

    doc_id = uuid.uuid4()
    chunks = [
        RetrievalResult(
            chunk_id=uuid.uuid4(),
            document_id=doc_id,
            text=sec["content"],
            score=0.95,
            metadata={"title": sec["title"], "source": "openapi.json"},
        )
        for sec in parsed.sections
    ]

    retriever = InMemoryTestRetriever(chunks)
    orchestrator = RAGOrchestratorService(
        retriever=retriever,
        context_builder=ContextBuilder(),
        generation_service=GenerationService(llm_provider=MockLLMProvider()),
        grounding_validator=GroundingValidator(),
        mandatory_grounding=True,
    )

    test_queries = [
        "What endpoints are available?",
        "What does the /health endpoint do?",
        "Summarize the API specification",
    ]

    forbidden_patterns = [
        "paths./",
        "paths.",
        "info.title",
        "get.tags",
        "get.summary",
        "get.operationId",
        "post.responses",
        "components.schemas.",
    ]

    for q in test_queries:
        res = await orchestrator.execute(query=q)
        assert res.grounded is True
        for pat in forbidden_patterns:
            assert pat not in res.answer, (
                f"Forbidden keypath pattern '{pat}' found in answer for query '{q}':\n{res.answer}"
            )


@pytest.mark.asyncio
async def test_markdown_heading_links_do_not_leak_into_answers() -> None:
    """TEST 12: Markdown heading and anchor links are normalized and do not leak into answers."""
    doc_content = (
        "# Project Overview\n\n"
        "Table of contents:\n"
        "- [System Architecture](#-system-architecture)\n"
        "- [Health & Readiness](#health-readiness)\n"
        "- [API Specs](-api-specs)\n\n"
        "## System Architecture\n"
        "The system architecture delivers multi-tenant retrieval isolation across all components.\n\n"
        "## Health & Readiness\n"
        "The GET /health endpoint provides service liveness telemetry."
    )

    doc_id = uuid.uuid4()
    chunk = RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        text=doc_content,
        score=0.95,
        metadata={"title": "README.md", "source": "README.md"},
    )

    retriever = InMemoryTestRetriever([chunk])
    orchestrator = RAGOrchestratorService(
        retriever=retriever,
        context_builder=ContextBuilder(),
        generation_service=GenerationService(llm_provider=MockLLMProvider()),
        grounding_validator=GroundingValidator(),
        mandatory_grounding=True,
    )

    response = await orchestrator.execute(query="Describe the system architecture")
    assert response.grounded is True
    assert "System Architecture" in response.answer or "system architecture" in response.answer.lower()

    # Must NOT leak internal Markdown navigation link syntax or anchor slugs
    assert "[System Architecture]" not in response.answer
    assert "(-system-architecture)" not in response.answer
    assert "(#-system-architecture)" not in response.answer
    assert "(-api-specs)" not in response.answer

    # Valid citations [1] and technical tokens remain intact
    assert len(response.citations) >= 1
    assert any(f"[{cid}]" in response.answer for cid in response.citations)


@pytest.mark.asyncio
async def test_external_links_and_technical_endpoints_handled_safely() -> None:
    """TEST 13: External links are normalized to readable text while technical endpoints remain intact."""
    doc_content = (
        "For more details, visit the official [Developer Portal](https://example.com/docs).\n"
        "The service exposes the GET /health and POST /api/v1/query endpoints for monitoring."
    )

    doc_id = uuid.uuid4()
    chunk = RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        text=doc_content,
        score=0.92,
        metadata={"title": "Integration Guide", "source": "guide.md"},
    )

    retriever = InMemoryTestRetriever([chunk])
    orchestrator = RAGOrchestratorService(
        retriever=retriever,
        context_builder=ContextBuilder(),
        generation_service=GenerationService(llm_provider=MockLLMProvider()),
        grounding_validator=GroundingValidator(),
        mandatory_grounding=True,
    )

    response = await orchestrator.execute(query="Where is the developer portal?")
    assert response.grounded is True
    # Label is preserved
    assert "Developer Portal" in response.answer or "developer portal" in response.answer.lower()
    # URL target is stripped from prose
    assert "https://example.com/docs" not in response.answer
    assert "[Developer Portal]" not in response.answer

    # Technical endpoints query
    ep_response = await orchestrator.execute(query="What endpoints are used for monitoring?")
    assert ep_response.grounded is True
    # Technical endpoint path is preserved intact
    assert "GET /health" in ep_response.answer or "/health" in ep_response.answer


@pytest.mark.asyncio
async def test_explicit_raw_content_request_preserves_markdown_links() -> None:
    """TEST 14: Explicit raw content request preserves raw Markdown and links verbatim."""
    raw_markdown = (
        "# System Architecture\n"
        "See [System Architecture](#-system-architecture) and [External API](https://example.com).\n"
        "Endpoint: GET /health"
    )

    doc_id = uuid.uuid4()
    chunk = RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        text=raw_markdown,
        score=0.95,
        metadata={"title": "raw_arch.md", "source": "raw_arch.md"},
    )

    retriever = InMemoryTestRetriever([chunk])
    orchestrator = RAGOrchestratorService(
        retriever=retriever,
        context_builder=ContextBuilder(),
        generation_service=GenerationService(llm_provider=MockLLMProvider()),
        grounding_validator=GroundingValidator(),
        mandatory_grounding=True,
    )

    response = await orchestrator.execute(query="Show me the raw content of the architecture document")
    assert response.grounded is True
    # Explicit raw request keeps verbatim Markdown links inside code block
    assert "```" in response.answer
    assert "[System Architecture](#-system-architecture)" in response.answer
    assert "[External API](https://example.com)" in response.answer


def test_citation_manifest_groups_identical_source_documents() -> None:
    """TEST 15: Citation manifest groups multiple chunks from the identical source document."""
    from app.rag.context.domain import AssembledContext, ContextDocument

    doc_id = uuid.uuid4()
    doc1 = ContextDocument(
        citation_id=1,
        citation_label="[1]",
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        source_id="README.md",
        title="README.md",
        page_number=None,
        score=0.91,
        content="Chunk 1 content",
        token_count=10,
    )
    doc2 = ContextDocument(
        citation_id=2,
        citation_label="[2]",
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        source_id="README.md",
        title="README.md",
        page_number=None,
        score=0.88,
        content="Chunk 2 content",
        token_count=10,
    )

    assembled = AssembledContext(
        formatted_context="...",
        documents=[doc1, doc2],
        citation_map={1: doc1, 2: doc2},
        total_tokens=20,
        total_chunks=2,
        truncated=False,
        dropped_chunks_count=0,
    )

    manifest = assembled.format_citation_manifest()
    # Should group [1] and [2] on one line rather than duplicate lines
    assert '[1] "README.md" [also [2]] (Source: README.md, Score: 0.910)' in manifest
    assert manifest.count("README.md") == 2  # Once in title, once in source, on 1 single line!


