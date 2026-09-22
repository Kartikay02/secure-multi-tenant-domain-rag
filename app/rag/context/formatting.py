"""Formatting and boundary preservation for context blocks injected into LLM prompts."""

from typing import Literal

from app.rag.context.domain import ContextDocument


class ContextFormatter:
    """Formats assembled context documents with unambiguous source boundaries and citation labels."""

    def __init__(self, style: Literal["markdown", "xml"] = "markdown") -> None:
        self.style = style

    def format_document(self, doc: ContextDocument) -> str:
        """Format a single ContextDocument into its delimited prompt string."""
        page_str = f" | Page: {doc.page_number}" if doc.page_number is not None else ""

        if self.style == "xml":
            page_attr = f' page="{doc.page_number}"' if doc.page_number is not None else ""
            return (
                f'<document citation="{doc.citation_label}" id="{doc.chunk_id}" '
                f'source="{doc.source_id}" title="{doc.title}"{page_attr}>\n'
                f"{doc.content}\n"
                f"</document>"
            )

        # Markdown style (default)
        return (
            f"--- Context Document {doc.citation_label} ---\n"
            f"Title: {doc.title} | Source: {doc.source_id}{page_str}\n"
            f"Content:\n{doc.content}\n"
        )

    def format_all(self, documents: list[ContextDocument]) -> str:
        """Format an entire list of ContextDocuments into a single prompt block."""
        if not documents:
            return ""

        formatted_blocks = [self.format_document(doc) for doc in documents]
        separator = "\n\n"
        return separator.join(formatted_blocks)
