"""Factory for resolving format-specific document parsers."""

from pathlib import Path

from app.core.exceptions import UnsupportedFileTypeError
from app.rag.ingestion.parsers.base import DocumentParserProtocol
from app.rag.ingestion.parsers.csv_parser import CSVParser
from app.rag.ingestion.parsers.docx import DocxParser
from app.rag.ingestion.parsers.json_parser import JSONParser
from app.rag.ingestion.parsers.markdown import MarkdownParser
from app.rag.ingestion.parsers.pdf import PDFParser
from app.rag.ingestion.parsers.text import TextParser


class ParserFactory:
    """Registry and factory resolving parsers by file extension."""

    def __init__(self) -> None:
        self._parsers: dict[str, DocumentParserProtocol] = {}
        # Register standard parsers
        self.register_parser(TextParser())
        self.register_parser(MarkdownParser())
        self.register_parser(PDFParser())
        self.register_parser(DocxParser())
        self.register_parser(CSVParser())
        self.register_parser(JSONParser())

    def register_parser(self, parser: DocumentParserProtocol) -> None:
        """Register a parser for its supported extensions."""
        for ext in parser.supported_extensions():
            self._parsers[ext.lower()] = parser

    def get_parser(self, filename: str) -> DocumentParserProtocol:
        """Retrieve the appropriate parser for a given filename or extension."""
        ext = Path(filename).suffix.lower()
        parser = self._parsers.get(ext)
        if not parser:
            raise UnsupportedFileTypeError(
                extension=ext or "unknown",
                mime_type="unknown",
                allowed=list(self._parsers.keys()),
            )
        return parser


# Singleton factory instance
default_parser_factory = ParserFactory()
