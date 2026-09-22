"""Mock cross-encoder reranker for testing and offline simulation."""

import asyncio
import re
from collections.abc import Sequence

from app.core.exceptions import RerankError, RerankTimeoutError
from app.rag.reranking.interfaces import RerankerProtocol
from app.rag.vector.domain import RetrievalResult


class MockReranker(RerankerProtocol):
    """Deterministic mock cross-encoder reranker.

    Simulates joint query-document relevance scoring using lexical overlap,
    exact phrase matching, and candidate score weighting. Supports simulating
    delays, timeouts, and provider errors for test coverage.
    """

    def __init__(
        self,
        model_name: str = "mock-reranker-v1",
        should_fail: bool = False,
        simulate_timeout: bool = False,
        delay: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.should_fail = should_fail
        self.simulate_timeout = simulate_timeout
        self.delay = delay

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Rerank candidate chunks deterministically.

        Args:
            query: The search query.
            candidates: Candidate chunks from prior retrieval.
            top_k: Maximum number of results to return.

        Returns:
            List of RetrievalResult objects sorted descending by rerank score.

        Raises:
            RerankTimeoutError: When simulate_timeout is enabled.
            RerankError: When should_fail is enabled.
        """
        if self.delay > 0.0:
            await asyncio.sleep(self.delay)

        if self.simulate_timeout:
            raise RerankTimeoutError(provider="mock", timeout_seconds=10.0)

        if self.should_fail:
            raise RerankError(
                "Mock reranking failure simulated.",
                details={"model": self.model_name, "candidates_count": len(candidates)},
            )

        if not candidates:
            return []

        raw_terms = [w.lower() for w in re.findall(r"\w+", query) if len(w) >= 2]
        _stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "were",
            "will",
            "with",
            "what",
            "which",
            "where",
            "when",
            "who",
            "how",
            "why",
            "this",
            "about",
        }
        query_terms = set(w for w in raw_terms if w not in _stopwords) or set(raw_terms)
        clean_query = query.strip().lower()

        scored_results: list[RetrievalResult] = []

        for cand in candidates:
            text_lower = cand.text.lower()
            doc_terms = set(re.findall(r"\w+", text_lower))

            # 1. Term overlap fraction
            if query_terms:
                matching_terms = query_terms.intersection(doc_terms)
                overlap_fraction = len(matching_terms) / len(query_terms)
            else:
                overlap_fraction = 0.0

            # 2. Exact phrase boost
            phrase_bonus = 0.25 if clean_query in text_lower and clean_query else 0.0

            # 3. Summary query intent boost for introductory chunks
            is_summary = any(
                kw in clean_query
                for kw in ["summarize", "summary", "overview", "what is this", "about this"]
            )
            summary_bonus = 0.0
            if is_summary:
                chunk_idx = cand.metadata.get("chunk_index")
                if chunk_idx == 0:
                    summary_bonus = 0.35
                elif any(
                    h in text_lower
                    for h in ["overview", "architecture", "consolidated", "introduction", "readme"]
                ):
                    summary_bonus = 0.20

            # 4. Incorporate prior candidate score as weak prior
            prior_weight = min(1.0, max(0.0, cand.score)) * 0.15

            # Joint simulated cross-encoder score
            raw_score = (0.60 * overlap_fraction) + phrase_bonus + summary_bonus + prior_weight
            calibrated_score = round(min(1.0, max(0.01, raw_score)), 4)

            scored_metadata = dict(cand.metadata)
            scored_metadata["reranker"] = self.model_name
            scored_metadata["rerank_raw_overlap"] = round(overlap_fraction, 4)

            scored_cand = RetrievalResult(
                chunk_id=cand.chunk_id,
                document_id=cand.document_id,
                text=cand.text,
                score=calibrated_score,
                dense_score=cand.dense_score,
                sparse_score=cand.sparse_score,
                rerank_score=calibrated_score,
                metadata=scored_metadata,
            )
            scored_results.append(scored_cand)

        # Sort descending by rerank_score (and secondary tie-break by original score)
        scored_results.sort(
            key=lambda r: (r.rerank_score if r.rerank_score is not None else 0.0, r.score),
            reverse=True,
        )

        return scored_results[:top_k]
