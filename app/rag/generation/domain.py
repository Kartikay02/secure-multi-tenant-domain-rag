"""Domain models and Pydantic schemas for LLM generation and structured outputs."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.rag.context.domain import ContextDocument


class GenerationParameters(BaseModel):
    """Configurable runtime parameters for LLM generation requests."""

    model_config = ConfigDict(frozen=True)

    temperature: float = Field(
        default=0.0, ge=0.0, le=2.0, description="Sampling temperature [0.0, 2.0]"
    )
    max_tokens: int = Field(default=1024, gt=0, description="Maximum completion tokens to generate")
    top_p: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Nucleus sampling probability threshold"
    )
    stop: list[str] | None = Field(default=None, description="Optional stop sequence tokens")
    stream: bool = Field(default=False, description="Whether to stream response tokens")


class StructuredGenerationPayload(BaseModel):
    """Pydantic schema enforcing structured JSON output from the LLM provider."""

    model_config = ConfigDict(extra="ignore")

    answer: str = Field(
        ...,
        description="Factual answer derived strictly and exclusively from the provided context chunks.",
    )
    citations: list[int] = Field(
        default_factory=list,
        description="Citation integer identifiers cited in the answer (e.g. [1, 2]).",
    )
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score [0.0, 1.0] reflecting evidentiary support.",
    )
    grounded: bool = Field(
        default=True,
        description="Flag indicating whether all claims are supported by the provided context.",
    )
    insufficient_context: bool = Field(
        default=False,
        description="Flag indicating whether the provided context was insufficient to answer the query.",
    )


class StreamChunk(BaseModel):
    """Incremental streaming token event emitted during generation."""

    model_config = ConfigDict(frozen=True)

    delta: str = Field(default="", description="Incremental text fragment")
    is_complete: bool = Field(default=False, description="True when stream transmission ends")
    finish_reason: str | None = Field(default=None, description="Completion termination reason")


class GeneratedResponse(BaseModel):
    """Domain response model encapsulating the generated answer and citation metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    answer: str
    citations: list[int] = Field(default_factory=list)
    confidence_score: float = 1.0
    grounded: bool = True
    insufficient_context: bool = False
    referenced_documents: list[ContextDocument] = Field(default_factory=list)
    model: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_seconds: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
