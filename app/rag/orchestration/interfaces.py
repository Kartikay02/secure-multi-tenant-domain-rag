"""Protocol contracts for RAG orchestration and generation service injection."""

from typing import Any, Protocol, runtime_checkable

from app.rag.context.domain import AssembledContext
from app.rag.generation.domain import GeneratedResponse, GenerationParameters
from app.rag.orchestration.domain import RAGResponse


@runtime_checkable
class GenerationServiceProtocol(Protocol):
    """Protocol for generation services translating query and assembled context into answers."""

    async def generate_answer(
        self,
        query: str,
        context: AssembledContext,
        parameters: GenerationParameters | None = None,
    ) -> GeneratedResponse:
        """Generate a structured, citation-mapped response from assembled context.

        Args:
            query: The user query string.
            context: AssembledContext containing formatted chunks and metadata.
            parameters: Optional generation hyperparameters.

        Returns:
            GeneratedResponse containing answer, citations, confidence, and metadata.
        """
        ...


@runtime_checkable
class RAGOrchestratorProtocol(Protocol):
    """Protocol for the primary RAG orchestrator coordinating all pipeline stages."""

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
    ) -> RAGResponse:
        """Coordinate query preprocessing, retrieval, reranking, context assembly,

        generation, grounding validation, and response construction.
        """
        ...
