"""Domain models for RAG orchestration requests, responses, and pipeline telemetry."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.rag.context.domain import ContextDocument
from app.rag.generation.domain import GenerationParameters


class PipelineStageLatency(BaseModel):
    """Execution latency breakdown across RAG pipeline stages in milliseconds."""

    model_config = ConfigDict(frozen=True)

    preprocessing_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    context_assembly_ms: float = 0.0
    generation_ms: float = 0.0
    validation_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert latencies to dictionary rounded to 2 decimal places."""
        return {
            "preprocessing_ms": round(self.preprocessing_ms, 2),
            "retrieval_ms": round(self.retrieval_ms, 2),
            "reranking_ms": round(self.reranking_ms, 2),
            "context_assembly_ms": round(self.context_assembly_ms, 2),
            "generation_ms": round(self.generation_ms, 2),
            "validation_ms": round(self.validation_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }


class RAGRequest(BaseModel):
    """Runtime request payload with optional overrides for pipeline stage hyperparameters."""

    model_config = ConfigDict(extra="ignore")

    query: str = Field(..., description="User search query string")
    request_id: str | None = Field(default=None, description="Optional caller correlation ID")
    filter_metadata: dict[str, Any] | None = Field(
        default=None, description="Metadata filtering constraints for retrieval"
    )
    retrieval_top_k: int | None = Field(
        default=None, gt=0, description="Override for candidate chunks retrieved"
    )
    score_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimum similarity cut-off score"
    )
    enable_reranking: bool = Field(
        default=True, description="Whether to execute cross-encoder reranking"
    )
    rerank_top_k: int | None = Field(
        default=None, gt=0, description="Override for top-k candidates after reranking"
    )
    max_context_tokens: int | None = Field(
        default=None, gt=0, description="Override for maximum tokens packed into context"
    )
    max_context_chunks: int | None = Field(
        default=None, gt=0, description="Override for maximum chunks included in context"
    )
    generation_parameters: GenerationParameters | None = Field(
        default=None, description="Hyperparameters for LLM answer generation"
    )
    enable_grounding: bool = Field(
        default=True, description="Whether to execute grounding validation guardrails"
    )
    grounding_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimum claim support ratio"
    )
    tenant_id: str | None = Field(
        default=None, description="Tenant identifier for multi-tenant isolation"
    )


class RAGResponse(BaseModel):
    """Complete domain-level response produced by the RAG orchestrator service."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str = Field(..., description="Preprocessed query string used across the pipeline")
    raw_query: str = Field(..., description="Original raw query passed by the caller")
    answer: str = Field(..., description="Final verified answer text")
    grounded: bool = Field(..., description="True if answer is substantiated by retrieved context")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Calibrated confidence score [0.0, 1.0]"
    )
    insufficient_context: bool = Field(
        default=False, description="True if context lacked sufficient evidence to answer"
    )
    fallback_applied: bool = Field(
        default=False, description="True if conservative fallback was executed"
    )
    citations: list[int] = Field(
        default_factory=list, description="Citation integer identifiers referenced in answer"
    )
    referenced_documents: list[ContextDocument] = Field(
        default_factory=list, description="Resolved source chunk context documents"
    )
    citation_manifest: str = Field(
        default="", description="Formatted references section mapping citations to sources"
    )
    unsupported_claims: list[str] = Field(
        default_factory=list, description="Claims flagged as lacking evidentiary support"
    )
    citation_errors: list[str] = Field(
        default_factory=list, description="Citation integrity error descriptions"
    )
    retrieved_chunks_count: int = Field(
        default=0, description="Number of candidate chunks retrieved from index"
    )
    context_chunks_count: int = Field(
        default=0, description="Number of chunks admitted into prompt context"
    )
    context_tokens: int = Field(default=0, description="Total token volume packed into context")
    stage_latencies: PipelineStageLatency = Field(
        ..., description="Latency breakdown across each pipeline stage"
    )
    request_id: str = Field(default="", description="Correlation or request tracking identifier")
    model: str = Field(
        default="", description="Identifier of the LLM model that generated the answer"
    )
    prompt_tokens: int | None = Field(default=None, description="Number of prompt tokens consumed")
    completion_tokens: int | None = Field(
        default=None, description="Number of completion tokens generated"
    )
    total_tokens: int | None = Field(
        default=None, description="Total token consumption for generation"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Pipeline diagnostic and telemetry metadata"
    )
