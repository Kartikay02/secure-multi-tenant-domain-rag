"""Domain models for assembled context and citation management."""

import re
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ContextDocument:
    """Individual document chunk assembled into prompt context with citation metadata."""

    citation_id: int
    citation_label: str  # e.g. "[1]"
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    source_id: str
    title: str
    page_number: int | None
    score: float
    content: str
    token_count: int
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AssembledContext:
    """Complete assembled context ready for injection into the LLM prompt."""

    formatted_context: str
    documents: list[ContextDocument]
    citation_map: dict[int, ContextDocument]
    total_tokens: int
    total_chunks: int
    truncated: bool
    dropped_chunks_count: int

    def get_citation(self, citation_id: int) -> ContextDocument | None:
        """Lookup a citation by integer identifier."""
        return self.citation_map.get(citation_id)

    def format_citation_manifest(self) -> str:
        """Generate a formatted reference manifest summarizing citations.

        Preserves chunk-level provenance internally while grouping identical
        source-document references in the user-facing citation panel.
        """
        if not self.documents:
            return "References: None"

        lines = ["References:"]
        grouped: dict[tuple[uuid.UUID, str, str], list[ContextDocument]] = {}
        for doc in self.documents:
            key = (doc.document_id, doc.title, doc.source_id)
            grouped.setdefault(key, []).append(doc)

        for (_doc_id, title, source_id), docs in grouped.items():
            if len(docs) == 1:
                doc = docs[0]
                page_info = f", Page {doc.page_number}" if doc.page_number is not None else ""
                lines.append(
                    f'{doc.citation_label} "{doc.title}"{page_info} '
                    f"(Source: {doc.source_id}, Score: {doc.score:.3f})"
                )
            else:
                other_labels = ", ".join(d.citation_label for d in docs[1:])
                pages = [d.page_number for d in docs if d.page_number is not None]
                if pages:
                    unique_pages = sorted(set(pages))
                    page_info = (
                        f", Pages {', '.join(str(p) for p in unique_pages)}"
                        if len(unique_pages) > 1
                        else f", Page {unique_pages[0]}"
                    )
                else:
                    page_info = ""
                best_score = max(d.score for d in docs)
                lines.append(
                    f'{docs[0].citation_label} "{title}"{page_info} [also {other_labels}] '
                    f"(Source: {source_id}, Score: {best_score:.3f})"
                )
        return "\n".join(lines)

    def resolve_citations(self, text: str) -> list[ContextDocument]:
        """Extract citations like [1], [2] from generated text and resolve to source chunks."""
        matches = re.findall(r"\[(\d+)\]", text)
        seen_ids: set[int] = set()
        resolved: list[ContextDocument] = []

        for m in matches:
            cid = int(m)
            if cid in self.citation_map and cid not in seen_ids:
                seen_ids.add(cid)
                resolved.append(self.citation_map[cid])

        return resolved

    @property
    def is_empty(self) -> bool:
        """True if no context documents were included."""
        return len(self.documents) == 0
