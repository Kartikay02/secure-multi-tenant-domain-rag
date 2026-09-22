"""Deterministic candidate deduplication using exact hashing and near-duplicate shingling."""

import hashlib
import re
from collections.abc import Sequence

from app.core.logging import get_logger
from app.rag.context.interfaces import DeduplicatorProtocol
from app.rag.vector.domain import RetrievalResult

logger = get_logger("app.rag.context.deduplication")


def _normalize_text(text: str) -> str:
    """Normalize text for hash comparison by collapsing whitespace and lowercasing."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _get_shingles(text: str, n: int = 1) -> set[str]:
    """Generate word shingles for near-duplicate Jaccard comparison."""
    words = re.findall(r"\w+", text.lower())
    if not words:
        return set()
    if n <= 1:
        return set(words)
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity coefficient between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


class ContentDeduplicator(DeduplicatorProtocol):
    """Prunes exact duplicates and near-duplicate chunks while retaining higher-scoring instances."""

    def __init__(self, default_threshold: float = 0.85) -> None:
        if not (0.0 <= default_threshold <= 1.0):
            raise ValueError(
                f"default_threshold must be between 0.0 and 1.0, got {default_threshold}"
            )
        self.default_threshold = default_threshold

    def deduplicate(
        self,
        candidates: Sequence[RetrievalResult],
        threshold: float | None = None,
    ) -> list[RetrievalResult]:
        """Remove exact and near-duplicate chunks from candidates.

        Args:
            candidates: Input candidate chunks.
            threshold: Optional threshold overriding instance default.

        Returns:
            Deduplicated candidate list preserving relative rank of kept items.
        """
        if not candidates:
            return []

        resolved_threshold = threshold if threshold is not None else self.default_threshold

        # Step 1: Exact hash deduplication
        # Map normalized hash to highest scoring candidate
        exact_seen_hashes: dict[str, RetrievalResult] = {}
        for cand in candidates:
            norm = _normalize_text(cand.text)
            if not norm:
                continue
            content_hash = hashlib.sha256(norm.encode("utf-8")).hexdigest()

            if content_hash not in exact_seen_hashes:
                exact_seen_hashes[content_hash] = cand
            else:
                existing = exact_seen_hashes[content_hash]
                # If current candidate has higher score, replace
                if cand.score > existing.score:
                    exact_seen_hashes[content_hash] = cand

        unique_candidates = list(exact_seen_hashes.values())

        # If threshold >= 1.0 or only 1 item, near-duplicate check is unnecessary
        if resolved_threshold >= 1.0 or len(unique_candidates) <= 1:
            return unique_candidates

        # Step 2: Near-duplicate suppression via Jaccard shingling
        # Sort by score descending so we process best candidates first
        sorted_by_score = sorted(unique_candidates, key=lambda c: c.score, reverse=True)
        kept_candidates: list[RetrievalResult] = []
        kept_shingles: list[set[str]] = []

        for cand in sorted_by_score:
            cand_shingles = _get_shingles(cand.text)
            is_duplicate = False

            for existing_shingles in kept_shingles:
                sim = _jaccard_similarity(cand_shingles, existing_shingles)
                if sim >= resolved_threshold:
                    is_duplicate = True
                    logger.debug(
                        f"Dropping near-duplicate chunk (sim={sim:.3f} >= {resolved_threshold}): "
                        f"'{cand.text[:40]}...'"
                    )
                    break

            if not is_duplicate:
                kept_candidates.append(cand)
                kept_shingles.append(cand_shingles)

        return kept_candidates
