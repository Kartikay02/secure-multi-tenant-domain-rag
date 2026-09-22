"""Intelligent chunk ordering strategies optimizing LLM attention and reading flow."""

from collections import deque

from app.rag.context.interfaces import OrderingStrategyProtocol
from app.rag.vector.domain import RetrievalResult


class RelevanceOrdering(OrderingStrategyProtocol):
    """Sorts candidates strictly descending by relevance score (standard greedy)."""

    def order(self, candidates: list[RetrievalResult]) -> list[RetrievalResult]:
        return sorted(candidates, key=lambda c: c.score, reverse=True)


class LostInTheMiddleOrdering(OrderingStrategyProtocol):
    """Mitigates LLM 'Lost in the Middle' attention degradation.

    Arranges highest relevance chunks at the prompt boundaries (beginning and end)
    where LLM attention and recall are mathematically and empirically highest,
    while placing lower-relevance chunks in the center.

    Order pattern: [Best (0), 3rd (2), 5th (4), ..., 6th (5), 4th (3), 2nd (1)]
    """

    def order(self, candidates: list[RetrievalResult]) -> list[RetrievalResult]:
        if len(candidates) <= 2:
            return sorted(candidates, key=lambda c: c.score, reverse=True)

        sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
        reordered: deque[RetrievalResult] = deque()

        # Place even indexed items on left, odd indexed items on right
        for idx, cand in enumerate(sorted_candidates):
            if idx % 2 == 0:
                reordered.appendleft(cand)  # Push to front
            else:
                reordered.append(cand)  # Push to back

        # Reverse so that index 0 is at the very beginning
        # e.g., for [c0, c1, c2, c3]:
        # idx 0 -> appendleft(c0) -> [c0]
        # idx 1 -> append(c1) -> [c0, c1]
        # idx 2 -> appendleft(c2) -> [c2, c0, c1]
        # idx 3 -> append(c3) -> [c2, c0, c1, c3]
        # Notice: c2 is start, c3 is end.
        # But we want c0 (best) at start, c1 (2nd best) at end!
        # Let's check alternating queue properly:
        # We can construct:
        # left = []
        # right = []
        # for i, c in enumerate(sorted_candidates):
        #     if i % 2 == 0:
        #         left.append(c)
        #     else:
        #         right.append(c)
        # return left + list(reversed(right))
        # Example for [c0, c1, c2, c3]:
        # left = [c0, c2]
        # right = [c1, c3]
        # reversed(right) = [c3, c1]
        # result = [c0, c2, c3, c1]
        # Start is c0 (rank 1), end is c1 (rank 2). In middle are c2 (rank 3) and c3 (rank 4)!
        # Exactly as described in Liu et al. (2023).

        left: list[RetrievalResult] = []
        right: list[RetrievalResult] = []
        for idx, cand in enumerate(sorted_candidates):
            if idx % 2 == 0:
                left.append(cand)
            else:
                right.append(cand)

        return left + list(reversed(right))


class DocumentOrderOrdering(OrderingStrategyProtocol):
    """Orders chunks by document_id and chunk_index to preserve document narrative flow."""

    def order(self, candidates: list[RetrievalResult]) -> list[RetrievalResult]:
        def sort_key(c: RetrievalResult) -> tuple[str, int]:
            doc_id = str(c.document_id)
            chunk_idx = int(c.metadata.get("chunk_index", 0))
            return (doc_id, chunk_idx)

        return sorted(candidates, key=sort_key)
