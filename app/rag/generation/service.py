import hashlib
from collections.abc import AsyncIterator

from app.core.logging import get_logger
from app.rag.context.domain import AssembledContext, ContextDocument
from app.rag.generation.domain import (
    GeneratedResponse,
    GenerationParameters,
    StructuredGenerationPayload,
)
from app.rag.generation.interfaces import LLMProviderProtocol, PromptBuilderProtocol
from app.rag.generation.prompt import PromptBuilder

logger = get_logger("app.rag.generation.service")


class GenerationService:
    """Orchestrates LLM prompt building, structured generation, and citation resolution."""

    def __init__(
        self,
        llm_provider: LLMProviderProtocol,
        prompt_builder: PromptBuilderProtocol | None = None,
        default_parameters: GenerationParameters | None = None,
    ) -> None:
        self.llm_provider = llm_provider
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.default_parameters = default_parameters or GenerationParameters()

    async def generate_answer(
        self,
        query: str,
        context: AssembledContext,
        parameters: GenerationParameters | None = None,
    ) -> GeneratedResponse:
        """Generate a grounded answer with verified citation mappings."""
        resolved_params = parameters or self.default_parameters

        # 1. Build Grounded Prompt
        system_prompt, user_prompt = self.prompt_builder.build_prompt(
            query=query,
            context=context,
        )

        query_fp = hashlib.sha256(query.encode("utf-8")).hexdigest()[:8]
        logger.info(
            f"Generating answer for query_fp={query_fp} (len={len(query)})",
            extra={
                "context_chunks": context.total_chunks,
                "context_tokens": context.total_tokens,
                "query_hash": query_fp,
                "query_length": len(query),
                "model": getattr(
                    self.llm_provider, "model", getattr(self.llm_provider, "model_name", "unknown")
                ),
            },
        )

        # 2. Call LLM Provider with Structured Schema
        response = await self.llm_provider.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            parameters=resolved_params,
            structured_schema=StructuredGenerationPayload,
        )

        # 3. Resolve Citations back to ContextDocuments
        # Combine explicitly declared citations with citations parsed from answer text
        text_resolved_docs = context.resolve_citations(response.answer)
        declared_citation_ids = set(response.citations)
        text_citation_ids = {doc.citation_id for doc in text_resolved_docs}
        combined_citation_ids = sorted(declared_citation_ids.union(text_citation_ids))

        referenced_docs: list[ContextDocument] = []
        for cid in combined_citation_ids:
            doc = context.get_citation(cid)
            if doc is not None:
                referenced_docs.append(doc)

        logger.info(
            f"Generation complete: {len(referenced_docs)} citations resolved, "
            f"grounded={response.grounded}, insufficient={response.insufficient_context}",
            extra={
                "citations": combined_citation_ids,
                "confidence": response.confidence_score,
                "latency_s": response.latency_seconds,
            },
        )

        # Return response with populated referenced_documents
        return GeneratedResponse(
            answer=response.answer,
            citations=combined_citation_ids,
            confidence_score=response.confidence_score,
            grounded=response.grounded,
            insufficient_context=response.insufficient_context,
            referenced_documents=referenced_docs,
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            latency_seconds=response.latency_seconds,
            metadata=response.metadata,
        )

    async def stream_answer(
        self,
        query: str,
        context: AssembledContext,
        parameters: GenerationParameters | None = None,
    ) -> AsyncIterator[str]:
        """Stream token deltas for real-time response generation."""
        resolved_params = parameters or self.default_parameters

        system_prompt, user_prompt = self.prompt_builder.build_prompt(
            query=query,
            context=context,
        )

        async for token in self.llm_provider.generate_stream(
            prompt=user_prompt,
            system_prompt=system_prompt,
            parameters=resolved_params,
        ):
            yield token
