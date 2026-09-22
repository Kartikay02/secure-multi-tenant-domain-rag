"""RAG query execution and Server-Sent Events (SSE) streaming endpoints."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse

from app.api.deps import (
    check_rate_limit,
    get_app_settings,
    get_rag_orchestrator,
    get_security_context,
    verify_api_key,
)
from app.core.config import Settings
from app.core.exceptions import AppException
from app.core.logging import get_logger
from app.rag.generation.domain import GenerationParameters
from app.schemas.query import QueryRequest, QueryResponse, ReferencedDocumentResponse
from app.security.authorization import SecurityContext
from app.services.rag_service import RAGOrchestratorService

logger = get_logger("app.api.v1.query")

router = APIRouter()

OrchestratorDep = Annotated[RAGOrchestratorService, Depends(get_rag_orchestrator)]
SettingsDep = Annotated[Settings, Depends(get_app_settings)]
SecurityContextDep = Annotated[SecurityContext, Depends(get_security_context)]


@router.post(
    "",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute grounded RAG query",
    description="Executes candidate retrieval, reranking, context assembly, LLM generation, and grounding verification.",
    dependencies=[Depends(check_rate_limit), Depends(verify_api_key)],
    responses={
        200: {"description": "Grounded answer with citation references and confidence score."},
        400: {"description": "Invalid query or malformed request parameters."},
        401: {"description": "Missing or unauthorized API key."},
        429: {"description": "Client rate limit exceeded."},
        504: {"description": "Query execution timed out."},
    },
)
async def execute_query(
    request: QueryRequest,
    orchestrator: OrchestratorDep,
    settings: SettingsDep,
    sec_ctx: SecurityContextDep,
) -> QueryResponse:
    """Execute end-to-end RAG query."""
    timeout_s = settings.llm.timeout_seconds

    # Construct generation parameters if temperature or max_tokens were overridden
    gen_params: GenerationParameters | None = None
    if request.temperature is not None or request.max_tokens is not None:
        gen_params = GenerationParameters(
            temperature=request.temperature
            if request.temperature is not None
            else settings.llm.temperature,
            max_tokens=request.max_tokens
            if request.max_tokens is not None
            else settings.llm.max_tokens,
        )

    try:
        async with asyncio.timeout(timeout_s):
            response = await orchestrator.execute(
                query=request.query,
                filter_metadata=request.filter_metadata,
                retrieval_top_k=request.top_k,
                score_threshold=request.score_threshold,
                enable_reranking=request.enable_reranking,
                rerank_top_k=request.rerank_top_k,
                max_context_tokens=request.max_context_tokens,
                generation_parameters=gen_params,
                enable_grounding=request.enable_grounding,
                grounding_threshold=request.grounding_threshold,
                tenant_id=sec_ctx.tenant_id,
                mode=request.mode,
            )
    except TimeoutError as exc:
        logger.error(
            f"Query execution timed out after {timeout_s}s (query_len={len(request.query)})"
        )
        raise AppException(
            message=f"Query execution timed out after {timeout_s} seconds.",
            error_code="QUERY_TIMEOUT",
            status_code=504,
            details={"timeout_seconds": timeout_s},
        ) from exc

    referenced_docs = [
        ReferencedDocumentResponse(
            citation_id=doc.citation_id,
            citation_label=doc.citation_label,
            chunk_id=doc.chunk_id,
            document_id=doc.document_id,
            source_id=doc.source_id,
            title=doc.title,
            page_number=doc.page_number,
            score=doc.score,
        )
        for doc in response.referenced_documents
    ]

    return QueryResponse(
        answer=response.answer,
        citations=response.citations,
        confidence_score=response.confidence_score,
        grounded=response.grounded,
        insufficient_context=response.insufficient_context,
        fallback_applied=response.fallback_applied,
        retrieval_metadata=response.metadata,
        request_id=response.request_id,
        citation_manifest=response.citation_manifest,
        referenced_documents=referenced_docs,
    )


@router.post(
    "/stream",
    status_code=status.HTTP_200_OK,
    summary="Stream grounded RAG query answer",
    description="Streams answer token deltas via Server-Sent Events (SSE), followed by completion metadata.",
    dependencies=[Depends(check_rate_limit), Depends(verify_api_key)],
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "Server-Sent Events streaming token chunks and final completion metadata.",
        },
        401: {"description": "Missing or unauthorized API key."},
        429: {"description": "Client rate limit exceeded."},
    },
)
async def stream_query(
    request: QueryRequest,
    orchestrator: OrchestratorDep,
    settings: SettingsDep,
    sec_ctx: SecurityContextDep,
) -> StreamingResponse:
    """Stream RAG response tokens incrementally via Server-Sent Events."""
    gen_params: GenerationParameters | None = None
    if request.temperature is not None or request.max_tokens is not None:
        gen_params = GenerationParameters(
            temperature=request.temperature
            if request.temperature is not None
            else settings.llm.temperature,
            max_tokens=request.max_tokens
            if request.max_tokens is not None
            else settings.llm.max_tokens,
        )

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for event in orchestrator.execute_stream(
                query=request.query,
                filter_metadata=request.filter_metadata,
                retrieval_top_k=request.top_k,
                score_threshold=request.score_threshold,
                enable_reranking=request.enable_reranking,
                rerank_top_k=request.rerank_top_k,
                max_context_tokens=request.max_context_tokens,
                generation_parameters=gen_params,
                enable_grounding=request.enable_grounding,
                grounding_threshold=request.grounding_threshold,
                tenant_id=sec_ctx.tenant_id,
                mode=request.mode,
            ):
                payload = json.dumps(event, default=str)
                yield f"data: {payload}\n\n"
        except asyncio.CancelledError:
            logger.info("Streaming client disconnected; generator cancelled.")
            raise
        except Exception as exc:
            logger.error(f"Error during query streaming: {exc}")
            error_payload = json.dumps(
                {"type": "error", "message": "An error occurred while streaming generation tokens."}
            )
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
