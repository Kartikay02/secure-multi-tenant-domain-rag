"""Dedicated RAG orchestration service coordinating query preprocessing, retrieval,

reranking, context construction, LLM generation, grounding validation, and telemetry.
"""

import hashlib
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from app.core.config import Settings, get_settings
from app.core.exceptions import PipelineStageError, QueryValidationError, SecurityValidationError
from app.core.logging import get_correlation_id, get_logger, set_correlation_id
from app.observability.hooks import get_telemetry_hook
from app.observability.interfaces import TelemetryHookProtocol
from app.rag.context.builder import ContextBuilder
from app.rag.context.domain import ContextDocument
from app.rag.context.interfaces import ContextBuilderProtocol
from app.rag.generation.domain import GenerationParameters
from app.rag.generation.interfaces import LLMProviderProtocol
from app.rag.generation.mock import MockLLMProvider
from app.rag.generation.service import GenerationService
from app.rag.orchestration.domain import PipelineStageLatency, RAGRequest, RAGResponse
from app.rag.orchestration.interfaces import (
    GenerationServiceProtocol,
    RAGOrchestratorProtocol,
)
from app.rag.query.classifier import QueryClassifier
from app.rag.query.expander import QueryExpander
from app.rag.query.interfaces import QueryPreprocessorProtocol
from app.rag.query.preprocessor import StandardQueryPreprocessor
from app.rag.query.router import QueryRouter
from app.rag.reranking.interfaces import RerankerProtocol
from app.rag.retrieval.interfaces import RetrieverProtocol
from app.rag.validation.interfaces import GroundingValidatorProtocol
from app.rag.validation.validator import GroundingValidator
from app.security.prompt_guard import PromptGuard

logger = get_logger("app.services.rag_service")


class RAGOrchestratorService(RAGOrchestratorProtocol):
    """Production-grade RAG orchestrator coordinating the multi-stage question-answering pipeline.

    Architecture:
        User Query
            ↓  [Stage 1: Query Preprocessing]
        Normalized Query
            ↓  [Stage 2: Candidate Retrieval]
        Retrieved Chunks
            ↓  [Stage 3: Cross-Encoder Reranking with Fallback]
        Reranked Chunks
            ↓  [Stage 4: Context Construction & Token Bounding]
        Assembled Context (with Citation Mapping)
            ↓  [Stage 5: LLM Answer Generation]
        Raw Generated Answer
            ↓  [Stage 6: Grounding Validation & Hallucination Guardrail]
        Verified Final Answer
            ↓  [Stage 7: Response Assembly & Citation Resolution]
        RAGResponse (Domain Model with Stage Latencies)
    """

    def __init__(
        self,
        retriever: RetrieverProtocol,
        context_builder: ContextBuilderProtocol,
        generation_service: GenerationServiceProtocol,
        reranker: RerankerProtocol | None = None,
        grounding_validator: GroundingValidatorProtocol | None = None,
        query_preprocessor: QueryPreprocessorProtocol | None = None,
        default_retrieval_top_k: int = 10,
        default_rerank_top_k: int = 5,
        default_enable_reranking: bool = True,
        default_max_context_tokens: int | None = None,
        default_max_context_chunks: int | None = None,
        default_generation_parameters: GenerationParameters | None = None,
        default_enable_grounding: bool = True,
        default_grounding_threshold: float | None = None,
        telemetry_hook: TelemetryHookProtocol | None = None,
        prompt_guard: PromptGuard | None = None,
        tenant_enforcement_enabled: bool = True,
        query_classifier: QueryClassifier | None = None,
        query_expander: QueryExpander | None = None,
        query_router: QueryRouter | None = None,
        mandatory_grounding: bool = True,
    ) -> None:
        self.retriever = retriever
        self.context_builder = context_builder
        self.generation_service = generation_service
        self.reranker = reranker
        self.grounding_validator = grounding_validator
        self.query_preprocessor = query_preprocessor or StandardQueryPreprocessor()
        self.query_classifier = query_classifier or QueryClassifier()
        self.query_expander = query_expander or QueryExpander()
        self.query_router = query_router or QueryRouter(
            default_retrieval_top_k=default_retrieval_top_k,
            default_rerank_top_k=default_rerank_top_k,
            default_score_threshold=default_grounding_threshold,
        )
        self.telemetry_hook = telemetry_hook or get_telemetry_hook()
        self.prompt_guard = prompt_guard
        self.tenant_enforcement_enabled = tenant_enforcement_enabled
        self.mandatory_grounding = mandatory_grounding

        self.default_retrieval_top_k = default_retrieval_top_k
        self.default_rerank_top_k = default_rerank_top_k
        self.default_enable_reranking = default_enable_reranking
        self.default_max_context_tokens = default_max_context_tokens
        self.default_max_context_chunks = default_max_context_chunks
        self.default_generation_parameters = default_generation_parameters or GenerationParameters()
        self.default_enable_grounding = default_enable_grounding
        self.default_grounding_threshold = default_grounding_threshold

        logger.info(
            "Initialized RAGOrchestratorService",
            extra={
                "retriever": self.retriever.__class__.__name__,
                "reranker": self.reranker.__class__.__name__ if self.reranker else "None",
                "context_builder": self.context_builder.__class__.__name__,
                "generation_service": self.generation_service.__class__.__name__,
                "grounding_validator": (
                    self.grounding_validator.__class__.__name__
                    if self.grounding_validator
                    else "None"
                ),
                "query_preprocessor": self.query_preprocessor.__class__.__name__,
                "prompt_guard": self.prompt_guard.__class__.__name__
                if self.prompt_guard
                else "None",
            },
        )

    async def execute_request(self, request: RAGRequest) -> RAGResponse:
        """Execute the RAG pipeline using a structured RAGRequest configuration object."""
        return await self.execute(
            query=request.query,
            request_id=request.request_id,
            filter_metadata=request.filter_metadata,
            retrieval_top_k=request.retrieval_top_k,
            score_threshold=request.score_threshold,
            enable_reranking=request.enable_reranking,
            rerank_top_k=request.rerank_top_k,
            max_context_tokens=request.max_context_tokens,
            max_context_chunks=request.max_context_chunks,
            generation_parameters=request.generation_parameters,
            enable_grounding=request.enable_grounding,
            grounding_threshold=request.grounding_threshold,
            tenant_id=request.tenant_id,
        )

    async def execute(
        self,
        query: str,
        request_id: str | None = None,
        filter_metadata: dict[str, Any] | None = None,
        retrieval_top_k: int | None = None,
        score_threshold: float | None = None,
        enable_reranking: bool | None = None,
        rerank_top_k: int | None = None,
        max_context_tokens: int | None = None,
        max_context_chunks: int | None = None,
        generation_parameters: GenerationParameters | None = None,
        enable_grounding: bool | None = None,
        grounding_threshold: float | None = None,
        tenant_id: str | None = None,
        mode: str = "rag",
    ) -> RAGResponse:
        """Execute the complete end-to-end RAG orchestration pipeline."""
        # --- Stage 0: Correlation & Telemetry Setup ---
        pipeline_start = time.perf_counter()
        correlation_id = request_id or get_correlation_id() or uuid.uuid4().hex
        set_correlation_id(correlation_id)

        query_fp = hashlib.sha256(str(query).encode("utf-8")).hexdigest()[:8]
        logger.info(
            f"Starting RAG orchestration for request {correlation_id}",
            extra={
                "request_id": correlation_id,
                "query_hash": query_fp,
                "query_length": len(query),
            },
        )

        pre_ms = 0.0
        ret_ms = 0.0
        rerank_ms = 0.0
        ctx_ms = 0.0
        gen_ms = 0.0
        val_ms = 0.0

        # --- Stage 1: Query Preprocessing & Security Scan ---
        t_pre = time.perf_counter()
        try:
            if self.query_preprocessor:
                clean_query = self.query_preprocessor.preprocess(query)
            else:
                clean_query = query.strip()
                if not clean_query:
                    raise QueryValidationError(
                        "Query must not be empty or whitespace-only.",
                        details={"query_length": 0},
                    )

            # Defensive Prompt Injection Scan
            if self.prompt_guard:
                self.prompt_guard.validate_or_raise(clean_query)
        except (QueryValidationError, SecurityValidationError):
            raise
        except Exception as exc:
            logger.error(f"[{correlation_id}] Query preprocessing failed: {exc}")
            raise PipelineStageError(
                stage="preprocessing",
                message=str(exc),
                request_id=correlation_id,
            ) from exc
        finally:
            pre_ms = (time.perf_counter() - t_pre) * 1000

        # --- Stage 1.5: Semantic Query Classification & Dynamic Parameter Routing ---
        intent_result = self.query_classifier.classify(clean_query)
        route_config = self.query_router.route(
            intent=intent_result.intent,
            override_top_k=retrieval_top_k,
            override_rerank_k=rerank_top_k,
            override_score_threshold=score_threshold,
        )

        # --- Stage 2: Candidate Retrieval with Tenant Isolation ---
        resolved_enable_rerank = (
            enable_reranking
            if enable_reranking is not None
            else (route_config.enable_reranking and self.default_enable_reranking)
        )
        resolved_rerank_top_k = rerank_top_k or route_config.rerank_top_k
        resolved_retrieval_top_k = retrieval_top_k or route_config.retrieval_top_k
        resolved_score_threshold = (
            score_threshold if score_threshold is not None else route_config.score_threshold
        )

        # If reranking is enabled and a reranker is configured, retrieve candidate pool
        if resolved_enable_rerank and self.reranker is not None:
            effective_retrieval_k = max(resolved_retrieval_top_k, resolved_rerank_top_k)
        else:
            effective_retrieval_k = resolved_retrieval_top_k

        # Inject tenant filter if tenant enforcement is active
        effective_filters: dict[str, Any] | None = None
        if filter_metadata:
            effective_filters = dict(filter_metadata)
        if self.tenant_enforcement_enabled and tenant_id:
            if effective_filters is None:
                effective_filters = {}
            # SEC-02: Force authenticated tenant to prevent tenant spoofing
            effective_filters["tenant_id"] = tenant_id

        t_ret = time.perf_counter()
        try:
            retrieved_chunks = await self.retriever.retrieve(
                query=clean_query,
                top_k=effective_retrieval_k,
                score_threshold=resolved_score_threshold,
                filter_metadata=effective_filters,
            )
        except Exception as exc:
            logger.error(f"[{correlation_id}] Candidate retrieval failed: {exc}")
            raise PipelineStageError(
                stage="retrieval",
                message=str(exc),
                request_id=correlation_id,
            ) from exc
        finally:
            ret_ms = (time.perf_counter() - t_ret) * 1000

        # SEC-02 defense-in-depth: prune any candidate chunk belonging to another tenant
        if self.tenant_enforcement_enabled and tenant_id:
            retrieved_chunks = [
                c
                for c in retrieved_chunks
                if not (
                    c.metadata
                    and c.metadata.get("tenant_id")
                    and str(c.metadata.get("tenant_id")) != str(tenant_id)
                )
            ]

        # --- Stage 3: Cross-Encoder Reranking with Graceful Fallback ---
        t_rerank = time.perf_counter()
        rerank_metadata: dict[str, Any] = {}
        candidates_for_context = retrieved_chunks

        if resolved_enable_rerank and self.reranker is not None and retrieved_chunks:
            try:
                candidates_for_context = await self.reranker.rerank(
                    query=clean_query,
                    candidates=retrieved_chunks,
                    top_k=resolved_rerank_top_k,
                )
                rerank_metadata["reranker_applied"] = True
                rerank_metadata["rerank_model"] = getattr(
                    self.reranker, "model", self.reranker.__class__.__name__
                )
            except Exception as exc:
                # Graceful fallback: log warning and proceed with retrieval top-k
                logger.warning(
                    f"[{correlation_id}] Reranking failed ({type(exc).__name__}: {exc}). "
                    f"Gracefully falling back to candidate retrieval results.",
                    extra={"request_id": correlation_id, "error": str(exc)},
                )
                candidates_for_context = retrieved_chunks[:resolved_rerank_top_k]
                rerank_metadata["rerank_fallback"] = True
                rerank_metadata["rerank_error"] = str(exc)
        else:
            candidates_for_context = retrieved_chunks[:resolved_rerank_top_k]
            rerank_metadata["reranker_applied"] = False

        rerank_ms = (time.perf_counter() - t_rerank) * 1000

        # --- Stage 4: Context Construction & Token Bounding ---
        resolved_max_tokens = max_context_tokens or self.default_max_context_tokens
        resolved_max_chunks = max_context_chunks or self.default_max_context_chunks

        t_ctx = time.perf_counter()
        try:
            assembled_context = self.context_builder.build_context(
                query=clean_query,
                candidates=candidates_for_context,
                max_tokens=resolved_max_tokens,
                max_chunks=resolved_max_chunks,
            )
        except Exception as exc:
            logger.error(f"[{correlation_id}] Context construction failed: {exc}")
            raise PipelineStageError(
                stage="context_assembly",
                message=str(exc),
                request_id=correlation_id,
            ) from exc
        finally:
            ctx_ms = (time.perf_counter() - t_ctx) * 1000

        # --- Stage 5: LLM Answer Generation ---
        resolved_gen_params = generation_parameters or self.default_generation_parameters

        t_gen = time.perf_counter()
        try:
            generated_response = await self.generation_service.generate_answer(
                query=clean_query,
                context=assembled_context,
                parameters=resolved_gen_params,
            )
        except Exception as exc:
            logger.error(f"[{correlation_id}] LLM generation failed: {exc}")
            raise PipelineStageError(
                stage="generation",
                message=str(exc),
                request_id=correlation_id,
            ) from exc
        finally:
            gen_ms = (time.perf_counter() - t_gen) * 1000

        # --- Stage 6: Grounding Validation & Guardrails ---
        if self.mandatory_grounding:
            if enable_grounding is False:
                logger.warning(
                    f"[{correlation_id}] Client attempted to disable mandatory grounding (enable_grounding=False); "
                    "overriding to True per server-enforced security policy."
                )
            resolved_enable_grounding = True
        else:
            resolved_enable_grounding = (
                enable_grounding if enable_grounding is not None else self.default_enable_grounding
            )

        final_answer = generated_response.answer
        fallback_applied = False
        unsupported_claims: list[str] = []
        citation_errors: list[str] = []

        # Grounding policy: empty context cannot be marked grounded
        if assembled_context.is_empty:
            grounded = False
            confidence_score = 0.0
            insufficient_context = True
        else:
            grounded = generated_response.grounded
            confidence_score = generated_response.confidence_score
            insufficient_context = generated_response.insufficient_context

        t_val = time.perf_counter()
        if resolved_enable_grounding and self.grounding_validator is not None:
            try:
                validation_result = self.grounding_validator.validate(
                    query=clean_query,
                    answer=generated_response.answer,
                    context=assembled_context,
                )
                final_answer = validation_result.final_answer
                grounded = validation_result.grounded
                confidence_score = validation_result.confidence_score
                insufficient_context = (
                    validation_result.insufficient_context
                    or generated_response.insufficient_context
                )
                fallback_applied = validation_result.fallback_applied
                unsupported_claims = validation_result.unsupported_claims
                citation_errors = validation_result.citation_errors
            except Exception as exc:
                logger.error(f"[{correlation_id}] Grounding validation failed: {exc}")
                raise PipelineStageError(
                    stage="grounding_validation",
                    message=str(exc),
                    request_id=correlation_id,
                ) from exc

        # Guarantee empty context is never marked grounded under any condition
        if assembled_context.is_empty:
            grounded = False
            confidence_score = 0.0
            insufficient_context = True

        val_ms = (time.perf_counter() - t_val) * 1000

        # --- Stage 7: Response Assembly & Citation Resolution ---
        total_ms = (time.perf_counter() - pipeline_start) * 1000

        stage_latencies = PipelineStageLatency(
            preprocessing_ms=pre_ms,
            retrieval_ms=ret_ms,
            reranking_ms=rerank_ms,
            context_assembly_ms=ctx_ms,
            generation_ms=gen_ms,
            validation_ms=val_ms,
            total_ms=total_ms,
        )

        # Resolve citations and referenced documents
        if fallback_applied:
            final_citations: list[int] = []
            referenced_docs: list[ContextDocument] = []
        else:
            text_resolved_docs = assembled_context.resolve_citations(final_answer)
            declared_cids = set(generated_response.citations)
            text_cids = {doc.citation_id for doc in text_resolved_docs}
            final_citations = sorted(declared_cids.union(text_cids))

            referenced_docs = []
            for cid in final_citations:
                doc = assembled_context.get_citation(cid)
                if doc is not None:
                    referenced_docs.append(doc)

        citation_manifest = assembled_context.format_citation_manifest()

        # Combine telemetry metadata
        merged_metadata: dict[str, Any] = {
            **generated_response.metadata,
            **rerank_metadata,
            "query_intent": intent_result.intent.value,
            "query_intent_confidence": intent_result.confidence,
            "query_intent_reason": intent_result.reason,
            "stage_latencies": stage_latencies.to_dict(),
            "context_truncated": assembled_context.truncated,
            "dropped_chunks_count": assembled_context.dropped_chunks_count,
        }

        conf_display = (
            f"{confidence_score:.2f}"
            if isinstance(confidence_score, (int, float))
            else str(confidence_score)
        )
        logger.info(
            f"[{correlation_id}] RAG pipeline complete in {total_ms:.2f}ms: "
            f"grounded={grounded}, confidence={conf_display}, fallback={fallback_applied}",
            extra={
                "request_id": correlation_id,
                "total_ms": round(total_ms, 2),
                "grounded": grounded,
                "confidence_score": confidence_score,
                "fallback_applied": fallback_applied,
                "citations_count": len(final_citations),
            },
        )

        response = RAGResponse(
            query=clean_query,
            raw_query=query,
            answer=final_answer,
            grounded=grounded,
            confidence_score=confidence_score,
            insufficient_context=insufficient_context,
            fallback_applied=fallback_applied,
            citations=final_citations,
            referenced_documents=referenced_docs,
            citation_manifest=citation_manifest,
            unsupported_claims=unsupported_claims,
            citation_errors=citation_errors,
            retrieved_chunks_count=len(retrieved_chunks),
            context_chunks_count=assembled_context.total_chunks,
            context_tokens=assembled_context.total_tokens,
            stage_latencies=stage_latencies,
            request_id=correlation_id,
            model=generated_response.model,
            prompt_tokens=generated_response.prompt_tokens,
            completion_tokens=generated_response.completion_tokens,
            total_tokens=generated_response.total_tokens,
            metadata=merged_metadata,
        )
        self.telemetry_hook.record_query_telemetry(response)
        return response

    async def execute_stream(
        self,
        query: str,
        request_id: str | None = None,
        filter_metadata: dict[str, Any] | None = None,
        retrieval_top_k: int | None = None,
        score_threshold: float | None = None,
        enable_reranking: bool | None = None,
        rerank_top_k: int | None = None,
        max_context_tokens: int | None = None,
        max_context_chunks: int | None = None,
        generation_parameters: GenerationParameters | None = None,
        enable_grounding: bool | None = None,
        grounding_threshold: float | None = None,
        tenant_id: str | None = None,
        mode: str = "rag",
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream answer tokens incrementally, followed by grounding verification and citation telemetry."""
        pipeline_start = time.perf_counter()
        correlation_id = request_id or get_correlation_id() or uuid.uuid4().hex
        set_correlation_id(correlation_id)

        pre_ms = 0.0
        ret_ms = 0.0
        rerank_ms = 0.0
        ctx_ms = 0.0
        gen_ms = 0.0
        val_ms = 0.0

        # Stage 1: Query Preprocessing & Security Scan
        t_pre = time.perf_counter()
        try:
            if self.query_preprocessor:
                clean_query = self.query_preprocessor.preprocess(query)
            else:
                clean_query = query.strip()
                if not clean_query:
                    raise QueryValidationError("Query must not be empty or whitespace-only.")

            if self.prompt_guard:
                self.prompt_guard.validate_or_raise(clean_query)
        except (QueryValidationError, SecurityValidationError):
            raise
        except Exception as exc:
            raise PipelineStageError(
                stage="preprocessing", message=str(exc), request_id=correlation_id
            ) from exc
        finally:
            pre_ms = (time.perf_counter() - t_pre) * 1000

        # Stage 1.5: Semantic Query Classification & Dynamic Parameter Routing
        intent_result = self.query_classifier.classify(clean_query)
        route_config = self.query_router.route(
            intent=intent_result.intent,
            override_top_k=retrieval_top_k,
            override_rerank_k=rerank_top_k,
            override_score_threshold=score_threshold,
        )

        # Stage 2: Candidate Retrieval with Tenant Isolation
        resolved_enable_rerank = (
            enable_reranking
            if enable_reranking is not None
            else (route_config.enable_reranking and self.default_enable_reranking)
        )
        resolved_rerank_top_k = rerank_top_k or route_config.rerank_top_k
        resolved_retrieval_top_k = retrieval_top_k or route_config.retrieval_top_k
        resolved_score_threshold = (
            score_threshold if score_threshold is not None else route_config.score_threshold
        )

        if resolved_enable_rerank and self.reranker is not None:
            effective_retrieval_k = max(resolved_retrieval_top_k, resolved_rerank_top_k)
        else:
            effective_retrieval_k = resolved_retrieval_top_k

        effective_filters: dict[str, Any] | None = None
        if filter_metadata:
            effective_filters = dict(filter_metadata)
        if self.tenant_enforcement_enabled and tenant_id:
            if effective_filters is None:
                effective_filters = {}
            # SEC-02: Force authenticated tenant to prevent tenant spoofing
            effective_filters["tenant_id"] = tenant_id

        t_ret = time.perf_counter()
        try:
            retrieved_chunks = await self.retriever.retrieve(
                query=clean_query,
                top_k=effective_retrieval_k,
                score_threshold=resolved_score_threshold,
                filter_metadata=effective_filters,
            )
        except Exception as exc:
            raise PipelineStageError(
                stage="retrieval", message=str(exc), request_id=correlation_id
            ) from exc
        finally:
            ret_ms = (time.perf_counter() - t_ret) * 1000

        # SEC-02 defense-in-depth: prune any candidate chunk belonging to another tenant
        if self.tenant_enforcement_enabled and tenant_id:
            retrieved_chunks = [
                c
                for c in retrieved_chunks
                if not (
                    c.metadata
                    and c.metadata.get("tenant_id")
                    and str(c.metadata.get("tenant_id")) != str(tenant_id)
                )
            ]

        # Stage 3: Cross-Encoder Reranking
        t_rerank = time.perf_counter()
        rerank_metadata: dict[str, Any] = {}
        candidates_for_context = retrieved_chunks

        if resolved_enable_rerank and self.reranker is not None and retrieved_chunks:
            try:
                candidates_for_context = await self.reranker.rerank(
                    query=clean_query,
                    candidates=retrieved_chunks,
                    top_k=resolved_rerank_top_k,
                )
                rerank_metadata["reranker_applied"] = True
            except Exception as exc:
                candidates_for_context = retrieved_chunks[:resolved_rerank_top_k]
                rerank_metadata["rerank_fallback"] = True
                rerank_metadata["rerank_error"] = str(exc)
        else:
            candidates_for_context = retrieved_chunks[:resolved_rerank_top_k]
            rerank_metadata["reranker_applied"] = False

        rerank_ms = (time.perf_counter() - t_rerank) * 1000

        # Stage 4: Context Construction
        resolved_max_tokens = max_context_tokens or self.default_max_context_tokens
        resolved_max_chunks = max_context_chunks or self.default_max_context_chunks

        t_ctx = time.perf_counter()
        try:
            assembled_context = self.context_builder.build_context(
                query=clean_query,
                candidates=candidates_for_context,
                max_tokens=resolved_max_tokens,
                max_chunks=resolved_max_chunks,
            )
        except Exception as exc:
            raise PipelineStageError(
                stage="context_assembly", message=str(exc), request_id=correlation_id
            ) from exc
        finally:
            ctx_ms = (time.perf_counter() - t_ctx) * 1000

        # Stage 5: Generation (Streamed)
        resolved_gen_params = generation_parameters or self.default_generation_parameters
        t_gen = time.perf_counter()
        accumulated_tokens: list[str] = []

        try:
            if hasattr(self.generation_service, "stream_answer"):
                async for token in self.generation_service.stream_answer(
                    query=clean_query,
                    context=assembled_context,
                    parameters=resolved_gen_params,
                ):
                    accumulated_tokens.append(token)
                    yield {"type": "token", "delta": token}
                accumulated_answer = "".join(accumulated_tokens).strip()
            else:
                resp = await self.generation_service.generate_answer(
                    query=clean_query,
                    context=assembled_context,
                    parameters=resolved_gen_params,
                )
                accumulated_answer = resp.answer
                yield {"type": "token", "delta": resp.answer}
        except Exception as exc:
            raise PipelineStageError(
                stage="generation", message=str(exc), request_id=correlation_id
            ) from exc
        finally:
            gen_ms = (time.perf_counter() - t_gen) * 1000

        # Stage 6: Grounding Validation & Guardrails
        if self.mandatory_grounding:
            if enable_grounding is False:
                logger.warning(
                    f"[{correlation_id}] Client attempted to disable mandatory grounding in stream; "
                    "overriding to True per server-enforced security policy."
                )
            resolved_enable_grounding = True
        else:
            resolved_enable_grounding = (
                enable_grounding if enable_grounding is not None else self.default_enable_grounding
            )
        final_answer = accumulated_answer
        if assembled_context.is_empty:
            grounded = False
            confidence_score = 0.0
            insufficient_context = True
        else:
            grounded = False
            confidence_score = 0.0
            insufficient_context = False
        fallback_applied = False

        t_val = time.perf_counter()
        if resolved_enable_grounding and self.grounding_validator is not None:
            try:
                val_res = self.grounding_validator.validate(
                    query=clean_query,
                    answer=accumulated_answer,
                    context=assembled_context,
                )
                final_answer = val_res.final_answer
                grounded = val_res.grounded
                confidence_score = val_res.confidence_score
                insufficient_context = val_res.insufficient_context
                fallback_applied = val_res.fallback_applied
                if fallback_applied:
                    yield {"type": "fallback", "answer": final_answer}
            except Exception as exc:
                raise PipelineStageError(
                    stage="grounding_validation", message=str(exc), request_id=correlation_id
                ) from exc

        if assembled_context.is_empty:
            grounded = False
            confidence_score = 0.0
            insufficient_context = True

        val_ms = (time.perf_counter() - t_val) * 1000

        # Stage 7: Final Metadata Event
        total_ms = (time.perf_counter() - pipeline_start) * 1000
        stage_latencies = PipelineStageLatency(
            preprocessing_ms=pre_ms,
            retrieval_ms=ret_ms,
            reranking_ms=rerank_ms,
            context_assembly_ms=ctx_ms,
            generation_ms=gen_ms,
            validation_ms=val_ms,
            total_ms=total_ms,
        )

        referenced_docs: list[ContextDocument] = []
        if fallback_applied:
            final_citations = []
        else:
            text_resolved_docs = assembled_context.resolve_citations(final_answer)
            final_citations = sorted({doc.citation_id for doc in text_resolved_docs})
            for cid in final_citations:
                doc = assembled_context.get_citation(cid)
                if doc is not None:
                    referenced_docs.append(doc)

        citation_manifest = assembled_context.format_citation_manifest()
        merged_metadata = {
            **rerank_metadata,
            "query_intent": intent_result.intent.value,
            "query_intent_confidence": intent_result.confidence,
            "query_intent_reason": intent_result.reason,
            "stage_latencies": stage_latencies.to_dict(),
            "context_truncated": assembled_context.truncated,
            "dropped_chunks_count": assembled_context.dropped_chunks_count,
        }

        yield {
            "type": "complete",
            "answer": final_answer,
            "citations": final_citations,
            "confidence_score": confidence_score,
            "grounded": grounded,
            "insufficient_context": insufficient_context,
            "fallback_applied": fallback_applied,
            "retrieval_metadata": merged_metadata,
            "request_id": correlation_id,
            "citation_manifest": citation_manifest,
            "referenced_documents": [
                {
                    "citation_id": doc.citation_id,
                    "citation_label": doc.citation_label,
                    "chunk_id": str(doc.chunk_id),
                    "document_id": str(doc.document_id),
                    "source_id": doc.source_id,
                    "title": doc.title,
                    "page_number": doc.page_number,
                    "score": doc.score,
                }
                for doc in referenced_docs
            ],
        }


def create_rag_orchestrator(
    retriever: RetrieverProtocol,
    generation_service: GenerationServiceProtocol | None = None,
    llm_provider: LLMProviderProtocol | None = None,
    reranker: RerankerProtocol | None = None,
    context_builder: ContextBuilderProtocol | None = None,
    grounding_validator: GroundingValidatorProtocol | None = None,
    query_preprocessor: QueryPreprocessorProtocol | None = None,
    telemetry_hook: TelemetryHookProtocol | None = None,
    settings: Settings | None = None,
) -> RAGOrchestratorService:
    """Factory creating a fully wired RAGOrchestratorService with production defaults."""
    app_settings = settings or get_settings()

    # 1. Query Preprocessor
    preprocessor = query_preprocessor or StandardQueryPreprocessor(
        min_length=app_settings.orchestration.min_query_length,
        max_length=app_settings.orchestration.max_query_length,
    )

    # 2. Context Builder
    builder = context_builder or ContextBuilder(settings=app_settings.context)

    # 3. Generation Service
    if generation_service is not None:
        gen_svc = generation_service
    else:
        provider = llm_provider or MockLLMProvider()
        gen_svc = GenerationService(
            llm_provider=provider,
            default_parameters=GenerationParameters(
                temperature=app_settings.llm.temperature,
                max_tokens=app_settings.llm.max_tokens,
            ),
        )

    # 4. Grounding Validator
    validator = grounding_validator
    if validator is None and app_settings.guardrails.enabled:
        validator = GroundingValidator(settings=app_settings.guardrails)

    # 5. Prompt Guard
    prompt_guard = None
    if app_settings.security.prompt_injection_detection_enabled:
        prompt_guard = PromptGuard(
            high_risk_threshold=app_settings.security.prompt_injection_threshold
        )

    mandatory_grounding = app_settings.guardrails.mandatory or (
        app_settings.app.env in ("production", "staging")
    )

    return RAGOrchestratorService(
        retriever=retriever,
        context_builder=builder,
        generation_service=gen_svc,
        reranker=reranker,
        grounding_validator=validator,
        query_preprocessor=preprocessor,
        telemetry_hook=telemetry_hook,
        prompt_guard=prompt_guard,
        tenant_enforcement_enabled=app_settings.security.tenant_enforcement_enabled,
        default_retrieval_top_k=app_settings.orchestration.retrieval_top_k,
        default_rerank_top_k=app_settings.orchestration.rerank_top_k,
        default_enable_reranking=app_settings.orchestration.enable_reranking,
        default_max_context_tokens=app_settings.context.max_tokens,
        default_max_context_chunks=app_settings.context.max_chunks,
        default_enable_grounding=app_settings.orchestration.enable_grounding,
        default_grounding_threshold=app_settings.guardrails.grounding_threshold,
        mandatory_grounding=mandatory_grounding,
    )
