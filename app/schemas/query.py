"""Pydantic schemas for RAG query requests, responses, and SSE streaming events."""

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    """Natural language query request payload with hyperparameter overrides."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural language question or search query",
        examples=["How does HNSW achieve sub-millisecond query latency?"],
    )
    top_k: int | None = Field(
        default=None,
        gt=0,
        le=50,
        description="Override for number of candidate chunks retrieved",
    )
    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score cut-off",
    )
    filter_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata filtering constraints",
    )
    enable_reranking: bool = Field(
        default=True,
        description="Whether to execute cross-encoder reranking",
    )
    rerank_top_k: int | None = Field(
        default=None,
        gt=0,
        le=20,
        description="Override for number of reranked chunks passed to context",
    )
    max_context_tokens: int | None = Field(
        default=None,
        gt=0,
        le=8000,
        description="Maximum token budget ceiling for assembled prompt context",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="LLM generation temperature",
    )
    max_tokens: int | None = Field(
        default=None,
        gt=0,
        le=4096,
        description="Maximum completion tokens to generate",
    )
    enable_grounding: bool = Field(
        default=True,
        description="Whether to execute grounding validation guardrails",
    )
    grounding_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override for minimum claim support ratio",
    )
    mode: str = Field(
        default="auto",
        description="Query mode: 'auto' (hybrid AI, answers anything from documents or general knowledge) or 'rag' (strictly answers from uploaded documents)",
    )


class ReferencedDocumentResponse(BaseModel):
    """Source context document referenced in an answer citation."""

    citation_id: int = Field(..., description="Integer citation identifier (e.g. 1 for [1])")
    citation_label: str = Field(..., description="Formatted bracket citation label (e.g. '[1]')")
    chunk_id: uuid.UUID = Field(..., description="Unique chunk UUID")
    document_id: uuid.UUID = Field(..., description="Parent document UUID")
    source_id: str = Field(..., description="Original document source path or filename")
    title: str = Field(..., description="Document or section title")
    page_number: int | None = Field(default=None, description="Page number if available")
    score: float = Field(..., description="Retrieval or reranking relevance score")


class QueryResponse(BaseModel):
    """Complete structured domain response for a grounded RAG query."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    answer: str = Field(..., description="Factually grounded answer derived from context")
    citations: list[int] = Field(
        default_factory=list,
        description="Citation integer identifiers cited in the answer (e.g. [1, 2])",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Calibrated evidentiary confidence score [0.0, 1.0]",
    )
    grounded: bool = Field(
        ...,
        description="True if all claims are substantiated by cited context",
    )
    insufficient_context: bool = Field(
        default=False,
        description="True if retrieved context lacked evidence to answer the query",
    )
    fallback_applied: bool = Field(
        default=False,
        description="True if conservative refusal fallback was applied",
    )
    retrieval_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Stage execution latencies, token consumption, and pipeline telemetry",
    )
    request_id: str = Field(..., description="Unique correlation and request tracking ID")
    citation_manifest: str = Field(
        default="",
        description="Formatted reference text mapping citations to sources",
    )
    referenced_documents: list[ReferencedDocumentResponse] = Field(
        default_factory=list,
        description="List of context document chunks cited by the response",
    )


class StreamTokenEvent(BaseModel):
    """Incremental streaming token event."""

    type: str = "token"
    delta: str = Field(..., description="Incremental text fragment")


class StreamFallbackEvent(BaseModel):
    """Streaming event emitted when conservative refusal fallback replaces answer."""

    type: str = "fallback"
    answer: str = Field(..., description="Conservative refusal answer text")


class StreamCompleteEvent(BaseModel):
    """Final streaming event containing completion metadata and citations."""

    type: str = "complete"
    answer: str
    citations: list[int] = Field(default_factory=list)
    confidence_score: float
    grounded: bool
    insufficient_context: bool
    fallback_applied: bool
    retrieval_metadata: dict[str, Any] = Field(default_factory=dict)
    request_id: str
    citation_manifest: str = ""
    referenced_documents: list[ReferencedDocumentResponse] = Field(default_factory=list)
