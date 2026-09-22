"""Chunking protocol and data transfer models for RAG text segmentation."""

from typing import Any, Protocol

from pydantic import BaseModel, Field


class ChunkPayload(BaseModel):
    """Deterministic chunk payload containing text, index, metrics, and propagated metadata."""

    content: str = Field(..., description="Cleaned text content of the chunk")
    chunk_index: int = Field(..., description="0-indexed position within the document version")
    token_count: int = Field(..., description="Number of tokens in chunk")
    char_count: int = Field(..., description="Number of characters in chunk")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Propagated section, page, document, and structural metadata",
    )


class ChunkerProtocol(Protocol):
    """Abstract interface for text and document chunking strategies."""

    def split_text(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        start_index: int = 0,
    ) -> list[ChunkPayload]:
        """Split a raw text string into chunks."""
        ...

    def split_sections(
        self,
        sections: list[dict[str, Any]],
        base_metadata: dict[str, Any],
    ) -> list[ChunkPayload]:
        """Split structured document sections while preserving section and page boundaries."""
        ...
