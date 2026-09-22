"""Production context builder assembling deduplicated, bounded, citation-ready context."""

from collections.abc import Sequence
from typing import Any

from app.core.config import ContextSettings, get_settings
from app.core.logging import get_logger
from app.core.tokenizer import get_cached_encoding
from app.rag.context.compression import (
    ExtractiveQueryCompression,
    NoOpCompression,
    WhitespaceNormalizerCompression,
)
from app.rag.context.deduplication import ContentDeduplicator
from app.rag.context.domain import AssembledContext, ContextDocument
from app.rag.context.formatting import ContextFormatter
from app.rag.context.interfaces import (
    CompressionStrategyProtocol,
    ContextBuilderProtocol,
    DeduplicatorProtocol,
    OrderingStrategyProtocol,
)
from app.rag.context.ordering import (
    DocumentOrderOrdering,
    LostInTheMiddleOrdering,
    RelevanceOrdering,
)
from app.rag.vector.domain import RetrievalResult

logger = get_logger("app.rag.context.builder")


def _resolve_page_number(metadata: dict[str, Any]) -> int | None:
    """Extract and sanitize page number from candidate chunk metadata."""
    raw_page = metadata.get("page_number") if "page_number" in metadata else metadata.get("page")
    if raw_page is None:
        return None
    try:
        return int(raw_page)
    except (ValueError, TypeError):
        return None


class ContextBuilder(ContextBuilderProtocol):
    """Transforms retrieval candidates into bounded, deduplicated, and citation-tagged prompt context.

    Features:
    - Accurate token budgeting with tiktoken
    - Exact and near-duplicate candidate suppression
    - Pluggable ordering (Relevance, Lost in the Middle, Document Order)
    - Pluggable compression (No-op, Whitespace, Extractive)
    - Structured citation identifiers [1], [2] with 1-to-1 chunk mappings
    - Guaranteed model context boundary adherence
    """

    def __init__(
        self,
        settings: ContextSettings | None = None,
        deduplicator: DeduplicatorProtocol | None = None,
        ordering_strategy: OrderingStrategyProtocol | None = None,
        compression_strategy: CompressionStrategyProtocol | None = None,
        formatter: ContextFormatter | None = None,
        max_tokens: int | None = None,
        max_chunks: int | None = None,
        encoding_name: str | None = None,
    ) -> None:
        ctx_settings = settings or get_settings().context

        self.max_tokens = max_tokens if max_tokens is not None else ctx_settings.max_tokens
        self.max_chunks = max_chunks if max_chunks is not None else ctx_settings.max_chunks
        self.encoding_name = encoding_name or ctx_settings.encoding_name
        self._tokenizer = get_cached_encoding(self.encoding_name)

        self.deduplicator = deduplicator or ContentDeduplicator(
            default_threshold=ctx_settings.deduplication_threshold
        )

        # Resolve ordering strategy
        if ordering_strategy is not None:
            self.ordering_strategy = ordering_strategy
        else:
            strategy_name = ctx_settings.ordering_strategy.lower()
            if strategy_name == "lost_in_the_middle":
                self.ordering_strategy = LostInTheMiddleOrdering()
            elif strategy_name == "document_order":
                self.ordering_strategy = DocumentOrderOrdering()
            else:
                self.ordering_strategy = RelevanceOrdering()

        # Resolve compression strategy
        if compression_strategy is not None:
            self.compression_strategy = compression_strategy
        else:
            comp_name = ctx_settings.compression_strategy.lower()
            if comp_name == "whitespace":
                self.compression_strategy = WhitespaceNormalizerCompression()
            elif comp_name == "extractive":
                self.compression_strategy = ExtractiveQueryCompression()
            else:
                self.compression_strategy = NoOpCompression()

        self.formatter = formatter or ContextFormatter(style=ctx_settings.format_style)

        logger.info(
            "Initialized ContextBuilder",
            extra={
                "max_tokens": self.max_tokens,
                "max_chunks": self.max_chunks,
                "encoding": self.encoding_name,
                "ordering": self.ordering_strategy.__class__.__name__,
                "compression": self.compression_strategy.__class__.__name__,
                "format_style": self.formatter.style,
            },
        )

    def build_context(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        max_tokens: int | None = None,
        max_chunks: int | None = None,
    ) -> AssembledContext:
        """Transform candidates into bounded, citation-mapped prompt context."""
        resolved_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        resolved_max_chunks = max_chunks if max_chunks is not None else self.max_chunks

        if not candidates:
            logger.info("ContextBuilder received empty candidate list; returning empty context.")
            return AssembledContext(
                formatted_context="",
                documents=[],
                citation_map={},
                total_tokens=0,
                total_chunks=0,
                truncated=False,
                dropped_chunks_count=0,
            )

        initial_count = len(candidates)

        # 1. Deduplication
        deduped = self.deduplicator.deduplicate(candidates)

        # 2. Reordering
        ordered = self.ordering_strategy.order(deduped)

        # 3. Packing & Boundary Enforcement
        accepted_docs: list[ContextDocument] = []
        current_tokens = 0
        separator = "\n\n"
        separator_tokens = len(self._tokenizer.encode(separator))

        for cand in ordered:
            if len(accepted_docs) >= resolved_max_chunks:
                break

            # Apply compression
            compressed_text = self.compression_strategy.compress(cand.text, query)
            if not compressed_text.strip():
                continue

            metadata = dict(cand.metadata)
            page_number = _resolve_page_number(metadata)
            source_id = str(
                metadata.get("source")
                or metadata.get("source_id")
                or metadata.get("document_name")
                or cand.document_id
            )
            title = str(
                metadata.get("document_name")
                or metadata.get("title")
                or metadata.get("section_title")
                or "Document"
            )
            score = cand.rerank_score if cand.rerank_score is not None else cand.score

            citation_id = len(accepted_docs) + 1
            citation_label = f"[{citation_id}]"

            content_tokens = len(self._tokenizer.encode(compressed_text))

            doc = ContextDocument(
                citation_id=citation_id,
                citation_label=citation_label,
                chunk_id=cand.chunk_id,
                document_id=cand.document_id,
                source_id=source_id,
                title=title,
                page_number=page_number,
                score=round(score, 4),
                content=compressed_text,
                token_count=content_tokens,
                dense_score=cand.dense_score,
                sparse_score=cand.sparse_score,
                rerank_score=cand.rerank_score,
                metadata=metadata,
            )

            # Format the candidate block and check against token budget
            formatted_block = self.formatter.format_document(doc)
            block_tokens = len(self._tokenizer.encode(formatted_block))
            added_sep_tokens = separator_tokens if accepted_docs else 0
            candidate_total_tokens = current_tokens + added_sep_tokens + block_tokens

            if candidate_total_tokens <= resolved_max_tokens:
                accepted_docs.append(doc)
                current_tokens = candidate_total_tokens
            else:
                # Exceeds max_tokens budget, stop packing
                break

        formatted_context = self.formatter.format_all(accepted_docs)
        actual_total_tokens = (
            len(self._tokenizer.encode(formatted_context)) if formatted_context else 0
        )
        citation_map = {doc.citation_id: doc for doc in accepted_docs}
        dropped_count = initial_count - len(accepted_docs)
        truncated = dropped_count > 0

        logger.info(
            f"Assembled context with {len(accepted_docs)} chunks ({actual_total_tokens} tokens)",
            extra={
                "initial_candidates": initial_count,
                "included_chunks": len(accepted_docs),
                "dropped_chunks": dropped_count,
                "total_tokens": actual_total_tokens,
                "max_tokens": resolved_max_tokens,
                "truncated": truncated,
            },
        )

        return AssembledContext(
            formatted_context=formatted_context,
            documents=accepted_docs,
            citation_map=citation_map,
            total_tokens=actual_total_tokens,
            total_chunks=len(accepted_docs),
            truncated=truncated,
            dropped_chunks_count=dropped_count,
        )
