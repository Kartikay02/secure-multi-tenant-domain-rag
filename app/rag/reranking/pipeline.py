import hashlib
import time
from typing import Any

from app.core.logging import get_logger
from app.rag.reranking.interfaces import RerankerProtocol
from app.rag.retrieval.interfaces import RetrieverProtocol
from app.rag.vector.domain import RetrievalResult

logger = get_logger("app.rag.reranking.pipeline")


class RerankingPipeline(RetrieverProtocol):
    """Orchestrates candidate generation, cross-encoder reranking, and graceful fallback.

    Architecture:
        Query
          ↓
        Retriever (e.g. HybridRetriever)  --> [candidates_k chunks]
          ↓
        Cross-Encoder Reranker           --> [final_top_k chunks with rerank_score]
          ↓ (on reranker exception)
        Graceful Fallback                --> [candidates[:final_top_k] with fallback metadata]
    """

    def __init__(
        self,
        retriever: RetrieverProtocol,
        reranker: RerankerProtocol,
        candidates_k: int = 20,
        top_k: int = 5,
    ) -> None:
        if candidates_k <= 0:
            raise ValueError(f"candidates_k must be positive, got {candidates_k}")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        self.retriever = retriever
        self.reranker = reranker
        self.candidates_k = candidates_k
        self.top_k = top_k

    async def search(
        self,
        query: str,
        candidates_k: int | None = None,
        top_k: int | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Execute search: candidate retrieval followed by cross-encoder reranking.

        Args:
            query: The user query string.
            candidates_k: Number of candidate chunks to retrieve for reranking.
            top_k: Number of final ranked chunks to return.
            filter_metadata: Optional metadata filter dict for retrieval stage.

        Returns:
            List of final ranked RetrievalResult objects.
        """
        if not query or not query.strip():
            return []

        resolved_candidates_k = candidates_k or self.candidates_k
        resolved_top_k = top_k or self.top_k

        # Candidate pool must be at least as large as the desired final top_k
        effective_candidates_k = max(resolved_candidates_k, resolved_top_k)

        query_fp = hashlib.sha256(query.encode("utf-8")).hexdigest()[:8]
        start_time = time.perf_counter()
        logger.info(
            f"Starting two-stage search for query_fp={query_fp} (len={len(query)})",
            extra={
                "candidates_k": effective_candidates_k,
                "top_k": resolved_top_k,
                "retriever": self.retriever.__class__.__name__,
                "reranker": self.reranker.__class__.__name__,
                "query_hash": query_fp,
                "query_length": len(query),
            },
        )

        # 1. Candidate Retrieval Stage
        retrieval_start = time.perf_counter()
        candidates = await self.retriever.retrieve(
            query=query,
            top_k=effective_candidates_k,
            filter_metadata=filter_metadata,
        )
        retrieval_latency = time.perf_counter() - retrieval_start

        if not candidates:
            logger.info(
                "No candidate chunks retrieved for query",
                extra={
                    "query_hash": query_fp,
                    "query_length": len(query),
                    "latency_s": round(retrieval_latency, 4),
                },
            )
            return []

        logger.debug(
            f"Retrieved {len(candidates)} candidates in {retrieval_latency:.3f}s",
            extra={"candidate_count": len(candidates), "latency_s": round(retrieval_latency, 4)},
        )

        # 2. Cross-Encoder Reranking Stage with Graceful Fallback
        rerank_start = time.perf_counter()
        try:
            reranked_results = await self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=resolved_top_k,
            )
            rerank_latency = time.perf_counter() - rerank_start
            total_latency = time.perf_counter() - start_time

            logger.info(
                f"Reranking completed successfully: {len(reranked_results)} results returned",
                extra={
                    "results_count": len(reranked_results),
                    "rerank_latency_s": round(rerank_latency, 4),
                    "total_latency_s": round(total_latency, 4),
                },
            )
            return reranked_results

        except Exception as exc:
            # Graceful Fallback: do NOT fail user request if reranker encounters downtime/timeout/error
            rerank_latency = time.perf_counter() - rerank_start
            total_latency = time.perf_counter() - start_time

            logger.warning(
                f"Reranking stage failed ({type(exc).__name__}: {exc}). "
                f"Gracefully falling back to hybrid candidate ranking.",
                extra={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "candidates_available": len(candidates),
                    "fallback_top_k": resolved_top_k,
                    "rerank_latency_s": round(rerank_latency, 4),
                    "total_latency_s": round(total_latency, 4),
                },
            )

            fallback_results: list[RetrievalResult] = []
            for cand in candidates[:resolved_top_k]:
                annotated_metadata = dict(cand.metadata)
                annotated_metadata["rerank_fallback"] = True
                annotated_metadata["rerank_error"] = str(exc)

                fallback_cand = RetrievalResult(
                    chunk_id=cand.chunk_id,
                    document_id=cand.document_id,
                    text=cand.text,
                    score=cand.score,
                    dense_score=cand.dense_score,
                    sparse_score=cand.sparse_score,
                    rerank_score=None,
                    metadata=annotated_metadata,
                )
                fallback_results.append(fallback_cand)

            return fallback_results

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Conform to RetrieverProtocol for plug-and-play composability."""
        results = await self.search(
            query=query,
            candidates_k=self.candidates_k,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )
        if score_threshold is not None:
            results = [r for r in results if r.score >= score_threshold]
        return results
