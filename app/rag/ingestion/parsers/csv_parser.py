"""Tabular CSV document parser preserving column schemas and row contexts."""

import csv
import io
from pathlib import Path

from app.core.exceptions import ParsingError
from app.rag.ingestion.parsers.base import DocumentParserProtocol, ParsedDocument


class CSVParser(DocumentParserProtocol):
    """Parses tabular CSV files into contextual text representations with metadata."""

    def supported_extensions(self) -> set[str]:
        return {".csv"}

    async def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        """Decode and parse CSV content into structured row representations."""
        text_content: str
        encoding = "utf-8"
        try:
            text_content = file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                encoding = "latin-1"
                text_content = file_bytes.decode("latin-1")
            except UnicodeDecodeError as exc:
                raise ParsingError(filename=filename, reason=f"CSV decoding failed: {exc}") from exc

        derived_title = Path(filename).stem.replace("_", " ").title()

        try:
            reader = csv.reader(io.StringIO(text_content))
            rows = list(reader)
        except Exception as exc:
            raise ParsingError(filename=filename, reason=f"Malformed CSV syntax: {exc}") from exc

        if not rows:
            return ParsedDocument(
                content="",
                title=derived_title,
                author=None,
                sections=[],
                metadata={
                    "format": "text/csv",
                    "encoding": encoding,
                    "row_count": 0,
                    "column_count": 0,
                    "columns": [],
                },
            )

        headers = [h.strip() for h in rows[0]]
        data_rows = rows[1:]

        # Build formatted contextual text
        lines: list[str] = [
            f"# {derived_title} (Tabular Dataset)",
            f"Columns: {', '.join(headers)}",
            f"Total Records: {len(data_rows)}",
            "",
            "## Records",
        ]

        sections: list[dict[str, str]] = []
        batch_lines: list[str] = []
        batch_size = 50

        for idx, row in enumerate(data_rows, start=1):
            row_fields = []
            for col_idx, col_name in enumerate(headers):
                val = row[col_idx].strip() if col_idx < len(row) else ""
                row_fields.append(f"{col_name}: {val}")
            record_str = f"Record {idx}: " + " | ".join(row_fields)
            lines.append(record_str)
            batch_lines.append(record_str)

            if idx % batch_size == 0 or idx == len(data_rows):
                section_title = (
                    f"{derived_title} (Records {max(1, idx - len(batch_lines) + 1)}-{idx})"
                )
                sections.append(
                    {
                        "title": section_title,
                        "content": "\n".join(batch_lines),
                    }
                )
                batch_lines = []

        full_content = "\n".join(lines)

        return ParsedDocument(
            content=full_content,
            title=derived_title,
            author=None,
            sections=sections,
            metadata={
                "format": "text/csv",
                "encoding": encoding,
                "row_count": len(data_rows),
                "column_count": len(headers),
                "columns": headers,
                "byte_size": len(file_bytes),
            },
        )
