"""Markdown document parser with section and header extraction."""

import re
from pathlib import Path

from app.core.exceptions import ParsingError
from app.rag.ingestion.parsers.base import DocumentParserProtocol, ParsedDocument


class MarkdownParser(DocumentParserProtocol):
    """Parses Markdown (.md) documents, extracting headers, sections, and YAML frontmatter."""

    def supported_extensions(self) -> set[str]:
        return {".md"}

    def _extract_frontmatter(self, text: str) -> tuple[dict[str, str], str]:
        """Extract basic YAML frontmatter between leading triple dashes if present."""
        frontmatter: dict[str, str] = {}
        body = text
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                raw_fm = parts[1]
                body = parts[2].strip()
                for line in raw_fm.strip().splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        frontmatter[k.strip().lower()] = v.strip().strip("\"'")
        return frontmatter, body

    def _extract_sections(self, body: str) -> list[dict[str, str]]:
        """Split markdown into sections based on markdown heading indicators (#, ##, ###)."""
        sections: list[dict[str, str]] = []
        lines = body.splitlines()

        current_title = "Introduction"
        current_lines: list[str] = []

        for line in lines:
            header_match = re.match(r"^(#{1,4})\s+(.+)$", line)
            if header_match:
                if current_lines:
                    sections.append(
                        {
                            "title": current_title,
                            "content": "\n".join(current_lines).strip(),
                        }
                    )
                    current_lines = []
                current_title = header_match.group(2).strip()
            else:
                current_lines.append(line)

        if current_lines:
            sections.append(
                {
                    "title": current_title,
                    "content": "\n".join(current_lines).strip(),
                }
            )

        return [s for s in sections if s["content"]]

    async def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        """Parse markdown file bytes into structured sections."""
        try:
            raw_text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                raw_text = file_bytes.decode("latin-1")
            except UnicodeDecodeError as exc:
                raise ParsingError(
                    filename=filename, reason=f"Failed to decode markdown: {exc}"
                ) from exc

        frontmatter, body = self._extract_frontmatter(raw_text)

        # Determine document title
        title = frontmatter.get("title")
        if not title:
            h1_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
            if h1_match:
                title = h1_match.group(1).strip()
            else:
                title = Path(filename).stem.replace("_", " ").title()

        author = frontmatter.get("author")
        sections = self._extract_sections(body)

        return ParsedDocument(
            content=body,
            title=title,
            author=author,
            sections=sections,
            metadata={
                "format": "text/markdown",
                "has_frontmatter": bool(frontmatter),
                "section_count": len(sections),
                **frontmatter,
            },
        )
