"""Protocol contracts for context assembly, deduplication, ordering, and compression."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from app.rag.context.domain import AssembledContext
from app.rag.vector.domain import RetrievalResult


@runtime_checkable
class DeduplicatorProtocol(Protocol):
    """Protocol for candidate chunk deduplication."""

    def deduplicate(
        self,
        candidates: Sequence[RetrievalResult],
        threshold: float = 0.85,
    ) -> list[RetrievalResult]:
        """Remove exact and near-duplicate chunks while preserving highest-scoring instances.

        Args:
            candidates: Candidate chunks from retrieval/reranking.
            threshold: Jaccard similarity threshold for near-duplicate identification.

        Returns:
            Deduplicated list of candidates.
        """
        ...


@runtime_checkable
class OrderingStrategyProtocol(Protocol):
    """Protocol for ordering candidate chunks prior to prompt injection."""

    def order(self, candidates: list[RetrievalResult]) -> list[RetrievalResult]:
        """Order candidate chunks to optimize LLM attention and reading flow.

        Args:
            candidates: Deduplicated candidate chunks.

        Returns:
            Reordered list of candidate chunks.
        """
        ...


@runtime_checkable
class CompressionStrategyProtocol(Protocol):
    """Protocol for chunk content compression and noise reduction."""

    def compress(self, text: str, query: str) -> str:
        """Compress or prune irrelevant sentences from chunk text.

        Args:
            text: Original chunk content.
            query: User search query.

        Returns:
            Compressed chunk text.
        """
        ...


@runtime_checkable
class ContextBuilderProtocol(Protocol):
    """Protocol for context assembly orchestrators."""

    def build_context(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        max_tokens: int | None = None,
        max_chunks: int | None = None,
    ) -> AssembledContext:
        """Transform retrieval candidates into safe, bounded, citation-annotated LLM context.

        Args:
            query: User search query.
            candidates: Retrieved candidate chunks.
            max_tokens: Optional token ceiling overriding default.
            max_chunks: Optional chunk ceiling overriding default.

        Returns:
            AssembledContext with formatted prompt text and citation mapping.
        """
        ...
