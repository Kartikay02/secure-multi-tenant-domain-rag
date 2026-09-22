"""Document parser implementations and factory."""

from app.rag.ingestion.parsers.base import DocumentParserProtocol, ParsedDocument
from app.rag.ingestion.parsers.csv_parser import CSVParser
from app.rag.ingestion.parsers.docx import DocxParser
from app.rag.ingestion.parsers.factory import ParserFactory, default_parser_factory
from app.rag.ingestion.parsers.json_parser import JSONParser
from app.rag.ingestion.parsers.markdown import MarkdownParser
from app.rag.ingestion.parsers.pdf import PDFParser
from app.rag.ingestion.parsers.text import TextParser

__all__ = [
    "CSVParser",
    "DocumentParserProtocol",
    "DocxParser",
    "JSONParser",
    "MarkdownParser",
    "PDFParser",
    "ParsedDocument",
    "ParserFactory",
    "TextParser",
    "default_parser_factory",
]
