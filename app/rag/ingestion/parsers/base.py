"""Parser protocol and data transfer objects for document extraction."""

from typing import Any, Protocol

from pydantic import BaseModel, Field


class ParsedDocument(BaseModel):
    """Extracted text and structural metadata from a parsed document."""

    content: str = Field(..., description="Raw text extracted from the document")
    title: str | None = Field(default=None, description="Document title if detected")
    author: str | None = Field(default=None, description="Document author if detected")
    sections: list[dict[str, str]] = Field(
        default_factory=list,
        description="Structured sections with titles and content",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Format-specific metadata (e.g., page_count, creation_date)",
    )


class DocumentParserProtocol(Protocol):
    """Abstract interface implemented by all document file format parsers."""

    async def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        """Parse raw file bytes into a structured ParsedDocument."""
        ...

    def supported_extensions(self) -> set[str]:
        """Return the set of lowercase file extensions supported by this parser."""
        ...
