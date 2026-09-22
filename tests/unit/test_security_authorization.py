import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.exceptions import ForbiddenError
from app.rag.context.domain import AssembledContext
from app.rag.generation.domain import GeneratedResponse
from app.rag.orchestration.domain import RAGRequest
from app.rag.vector.domain import RetrievalResult
from app.security.authorization import (
    DocumentAction,
    SecurityContext,
    TenantDocumentAccessControl,
)
from app.services.rag_service import RAGOrchestratorService


def test_tenant_access_allowed_for_matching_tenant() -> None:
    """Verify access is allowed when document tenant matches caller tenant."""
    policy = TenantDocumentAccessControl(tenant_enforcement_enabled=True)
    ctx = SecurityContext(user_id="user_1", tenant_id="tenant_alpha", roles=("user",))
    doc_meta = {"tenant_id": "tenant_alpha", "category": "engineering"}

    assert policy.authorize_document(ctx, doc_meta, DocumentAction.READ) is True
    assert policy.authorize_document(ctx, doc_meta, DocumentAction.PROCESS) is True
    assert policy.authorize_document(ctx, doc_meta, DocumentAction.DELETE) is True


def test_cross_tenant_access_denied() -> None:
    """Verify cross-tenant document access raises ForbiddenError."""
    policy = TenantDocumentAccessControl(tenant_enforcement_enabled=True)
    ctx = SecurityContext(user_id="user_attacker", tenant_id="tenant_alpha", roles=("user",))
    victim_doc_meta = {"tenant_id": "tenant_beta", "category": "confidential"}

    for action in [
        DocumentAction.READ,
        DocumentAction.WRITE,
        DocumentAction.PROCESS,
        DocumentAction.DELETE,
    ]:
        with pytest.raises(ForbiddenError) as exc:
            policy.authorize_document(ctx, victim_doc_meta, action)
        assert "Access denied: Document belongs to tenant 'tenant_beta'" in str(exc.value)


def test_admin_role_can_bypass_tenant_boundaries() -> None:
    """Verify administrator and system roles can access documents across all tenants."""
    policy = TenantDocumentAccessControl(tenant_enforcement_enabled=True)
    admin_ctx = SecurityContext(user_id="superadmin", tenant_id="tenant_alpha", roles=("admin",))
    system_ctx = SecurityContext(user_id="cron_daemon", tenant_id="system", roles=("system",))
    tenant_b_doc = {"tenant_id": "tenant_beta"}

    assert policy.authorize_document(admin_ctx, tenant_b_doc, DocumentAction.READ) is True
    assert policy.authorize_document(admin_ctx, tenant_b_doc, DocumentAction.DELETE) is True
    assert policy.authorize_document(system_ctx, tenant_b_doc, DocumentAction.PROCESS) is True


def test_tenant_enforcement_disabled_allows_access() -> None:
    """Verify access is allowed unconditionally when tenant enforcement is toggled off."""
    policy = TenantDocumentAccessControl(tenant_enforcement_enabled=False)
    ctx = SecurityContext(user_id="user_1", tenant_id="tenant_alpha", roles=("user",))
    doc_meta = {"tenant_id": "tenant_beta"}

    assert policy.authorize_document(ctx, doc_meta, DocumentAction.READ) is True


@pytest.mark.asyncio
async def test_rag_orchestrator_injects_tenant_filter() -> None:
    """Verify RAGOrchestratorService automatically binds caller tenant_id to retriever filters."""
    mock_retriever = AsyncMock()
    mock_retriever.retrieve = AsyncMock(return_value=[])

    mock_builder = MagicMock()
    mock_builder.build_context.return_value = AssembledContext(
        formatted_context="",
        documents=[],
        citation_map={},
        total_tokens=0,
        total_chunks=0,
        truncated=False,
        dropped_chunks_count=0,
    )

    mock_gen_svc = AsyncMock()
    mock_gen_svc.generate_answer = AsyncMock(
        return_value=GeneratedResponse(
            answer="I do not have sufficient information.",
            citations=[],
            confidence_score=0.0,
            grounded=False,
            insufficient_context=True,
            model="mock",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
    )

    orchestrator = RAGOrchestratorService(
        retriever=mock_retriever,
        context_builder=mock_builder,
        generation_service=mock_gen_svc,
        reranker=None,
        grounding_validator=None,
        tenant_enforcement_enabled=True,
    )

    # Execute request with tenant_id and pre-existing filter metadata
    req = RAGRequest(
        query="What is the cluster topology?",
        filter_metadata={"department": "infra"},
        tenant_id="tenant_acme",
    )
    await orchestrator.execute_request(req)

    # Verify that retriever was called with combined tenant_id filter
    mock_retriever.retrieve.assert_called_once()
    call_kwargs = mock_retriever.retrieve.call_args.kwargs
    assert call_kwargs["filter_metadata"] == {
        "department": "infra",
        "tenant_id": "tenant_acme",
    }


@pytest.mark.asyncio
async def test_rag_orchestrator_defense_in_depth_prunes_cross_tenant_chunks() -> None:
    """Verify that even if a buggy or mock retriever returns chunks from another tenant,
    the orchestrator prunes them before context building or generation.
    """
    valid_chunk = RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        text="Authorized chunk for tenant_acme",
        score=0.9,
        metadata={"tenant_id": "tenant_acme"},
    )
    leaked_chunk = RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        text="CONFIDENTIAL chunk for tenant_evil",
        score=0.95,
        metadata={"tenant_id": "tenant_evil"},
    )

    mock_retriever = AsyncMock()
    # Retriever accidentally returns both chunks
    mock_retriever.retrieve = AsyncMock(return_value=[leaked_chunk, valid_chunk])

    mock_builder = MagicMock()
    mock_builder.build_context.return_value = AssembledContext(
        formatted_context="Authorized chunk for tenant_acme",
        documents=[],
        citation_map={},
        total_tokens=10,
        total_chunks=1,
        truncated=False,
        dropped_chunks_count=0,
    )

    mock_gen_svc = AsyncMock()
    mock_gen_svc.generate_answer = AsyncMock(
        return_value=GeneratedResponse(
            answer="Safe answer.",
            citations=[],
            confidence_score=0.9,
            grounded=True,
            insufficient_context=False,
            model="mock",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
    )

    orchestrator = RAGOrchestratorService(
        retriever=mock_retriever,
        context_builder=mock_builder,
        generation_service=mock_gen_svc,
        reranker=None,
        grounding_validator=None,
        tenant_enforcement_enabled=True,
    )

    await orchestrator.execute(
        query="Tell me about infrastructure",
        tenant_id="tenant_acme",
    )

    # Verify context builder was called ONLY with the valid tenant_acme chunk, leaked_chunk pruned!
    mock_builder.build_context.assert_called_once()
    candidates_passed = mock_builder.build_context.call_args.kwargs["candidates"]
    assert len(candidates_passed) == 1
    assert candidates_passed[0].text == "Authorized chunk for tenant_acme"
    assert candidates_passed[0].metadata["tenant_id"] == "tenant_acme"
