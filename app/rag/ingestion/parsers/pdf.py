"""PDF document parser using pypdf."""

import io
from pathlib import Path

import pypdf
from pypdf.errors import PdfReadError

from app.core.exceptions import ParsingError
from app.rag.ingestion.parsers.base import DocumentParserProtocol, ParsedDocument


class PDFParser(DocumentParserProtocol):
    """Extracts text, page boundaries, and document metadata from PDF files."""

    def supported_extensions(self) -> set[str]:
        return {".pdf"}

    async def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        """Parse PDF bytes into page sections and document metadata."""
        try:
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        except PdfReadError as exc:
            raise ParsingError(
                filename=filename, reason=f"Corrupted or invalid PDF format: {exc}"
            ) from exc
        except Exception as exc:
            raise ParsingError(filename=filename, reason=f"Failed to open PDF: {exc}") from exc

        if reader.is_encrypted:
            # Check if empty password works, otherwise raise error
            try:
                decrypted = reader.decrypt("")
                if decrypted == pypdf.PasswordType.NOT_DECRYPTED:
                    raise ParsingError(
                        filename=filename, reason="PDF is encrypted and requires a password."
                    )
            except Exception as exc:
                raise ParsingError(
                    filename=filename, reason=f"Password-protected PDF cannot be read: {exc}"
                ) from exc

        sections: list[dict[str, str]] = []
        full_content_parts: list[str] = []

        total_pages = len(reader.pages)
        if total_pages == 0:
            raise ParsingError(filename=filename, reason="PDF contains zero pages.")

        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:
                page_text = f"[Extraction error on page {page_idx}: {exc}]"

            cleaned_page = page_text.strip()
            if cleaned_page:
                sections.append(
                    {
                        "title": f"Page {page_idx}",
                        "content": cleaned_page,
                    }
                )
                full_content_parts.append(cleaned_page)

        # Extract standard PDF metadata fields
        meta = reader.metadata
        title = getattr(meta, "title", None) or Path(filename).stem.replace("_", " ").title()
        author = getattr(meta, "author", None)
        producer = str(getattr(meta, "producer", "") or "")
        creation_date_val = getattr(meta, "creation_date", None)
        creation_date = str(creation_date_val) if creation_date_val is not None else ""

        combined_content = "\n\n".join(full_content_parts)

        return ParsedDocument(
            content=combined_content,
            title=title,
            author=author,
            sections=sections,
            metadata={
                "format": "application/pdf",
                "page_count": total_pages,
                "producer": producer,
                "creation_date": creation_date,
            },
        )
