"""Plain text document parser."""

from pathlib import Path

from app.core.exceptions import ParsingError
from app.rag.ingestion.parsers.base import DocumentParserProtocol, ParsedDocument


class TextParser(DocumentParserProtocol):
    """Parses plain text (.txt) documents with encoding fallbacks."""

    def supported_extensions(self) -> set[str]:
        return {".txt"}

    async def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        """Decode plain text bytes and extract metadata."""
        content: str
        encoding = "utf-8"
        try:
            content = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                encoding = "latin-1"
                content = file_bytes.decode("latin-1")
            except UnicodeDecodeError as exc:
                raise ParsingError(
                    filename=filename, reason=f"Text decoding failed: {exc}"
                ) from exc

        derived_title = Path(filename).stem.replace("_", " ").title()

        return ParsedDocument(
            content=content,
            title=derived_title,
            author=None,
            sections=[{"title": derived_title, "content": content}],
            metadata={
                "format": "text/plain",
                "encoding": encoding,
                "byte_size": len(file_bytes),
            },
        )
