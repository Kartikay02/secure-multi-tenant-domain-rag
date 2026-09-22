"""Hybrid retriever combining dense and lexical search with result fusion."""

import asyncio
import hashlib
import time
from typing import Any

from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.rag.retrieval.fusion import ReciprocalRankFusion
from app.rag.retrieval.interfaces import FusionStrategyProtocol, RetrieverProtocol
from app.rag.vector.domain import RetrievalResult

logger = get_logger("app.rag.retrieval.hybrid")


def _get_underlying_session(retriever: Any) -> Any:
    """Extract underlying AsyncSession if available on retriever or its vector store."""
    if hasattr(retriever, "session"):
        try:
            return retriever.session
        except Exception:
            return None
    if hasattr(retriever, "vector_store") and hasattr(retriever.vector_store, "session"):
        try:
            return retriever.vector_store.session
        except Exception:
            return None
    return None


class HybridRetriever(RetrieverProtocol):
    """Orchestrates multi-modal retrieval combining dense vector and lexical search."""

    def __init__(
        self,
        dense_retriever: RetrieverProtocol,
        lexical_retriever: RetrieverProtocol,
        fusion_strategy: FusionStrategyProtocol | None = None,
        dense_top_k: int = 30,
        sparse_top_k: int = 30,
        final_top_k: int = 5,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.lexical_retriever = lexical_retriever
        self.fusion_strategy = fusion_strategy or ReciprocalRankFusion(k=60)
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.final_top_k = final_top_k

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Execute concurrent dense and lexical retrieval, followed by fusion and ranking."""
        if not query or not query.strip():
            return []

        clean_query = query.strip()
        resolved_final_top_k = top_k if top_k is not None else self.final_top_k
        start_time = time.perf_counter()

        query_fp = hashlib.sha256(clean_query.encode("utf-8")).hexdigest()[:8]
        query_len = len(clean_query)

        logger.info(
            f"Starting concurrent hybrid retrieval for query_fp={query_fp} (len={query_len})",
            extra={
                "dense_top_k": self.dense_top_k,
                "sparse_top_k": self.sparse_top_k,
                "final_top_k": resolved_final_top_k,
                "fusion_strategy": self.fusion_strategy.__class__.__name__,
                "query_hash": query_fp,
                "query_length": query_len,
            },
        )

        dense_session = _get_underlying_session(self.dense_retriever)
        lexical_session = _get_underlying_session(self.lexical_retriever)
        shares_session = (
            dense_session is not None
            and lexical_session is not None
            and dense_session is lexical_session
        )

        session_lock: asyncio.Lock | None = None
        if shares_session:
            # Single shared AsyncSession cannot execute overlapping queries simultaneously.
            # We serialize DB access on the shared session while embedding generation remains concurrent.
            session_lock = getattr(dense_session, "_hybrid_session_lock", None)
            if session_lock is None:
                session_lock = asyncio.Lock()
                dense_session._hybrid_session_lock = session_lock

        async def _run_dense() -> list[RetrievalResult]:
            if session_lock is not None:
                if hasattr(self.dense_retriever, "embedding_provider") and hasattr(
                    self.dense_retriever, "vector_store"
                ):
                    # Compute vector representation asynchronously (no DB access)
                    query_vector = await self.dense_retriever.embedding_provider.embed_text(
                        clean_query
                    )
                    # Protect DB query on shared session
                    async with session_lock:
                        raw_results = await self.dense_retriever.vector_store.similarity_search(
                            query_vector=query_vector,
                            top_k=self.dense_top_k,
                            score_threshold=score_threshold,
                            filter_metadata=filter_metadata,
                        )
                    return [
                        RetrievalResult(
                            chunk_id=r.chunk_id,
                            document_id=r.document_id,
                            text=r.text,
                            score=r.score,
                            dense_score=r.score,
                            sparse_score=r.sparse_score,
                            metadata=r.metadata,
                        )
                        for r in raw_results
                    ]
                else:
                    async with session_lock:
                        return await self.dense_retriever.retrieve(
                            query=clean_query,
                            top_k=self.dense_top_k,
                            score_threshold=score_threshold,
                            filter_metadata=filter_metadata,
                        )
            else:
                return await self.dense_retriever.retrieve(
                    query=clean_query,
                    top_k=self.dense_top_k,
                    score_threshold=score_threshold,
                    filter_metadata=filter_metadata,
                )

        async def _run_lexical() -> list[RetrievalResult]:
            if session_lock is not None:
                async with session_lock:
                    return await self.lexical_retriever.retrieve(
                        query=clean_query,
                        top_k=self.sparse_top_k,
                        score_threshold=score_threshold,
                        filter_metadata=filter_metadata,
                    )
            else:
                return await self.lexical_retriever.retrieve(
                    query=clean_query,
                    top_k=self.sparse_top_k,
                    score_threshold=score_threshold,
                    filter_metadata=filter_metadata,
                )

        # 1. Execute concurrent retrieval using asyncio.gather
        results = await asyncio.gather(_run_dense(), _run_lexical(), return_exceptions=True)
        dense_res, sparse_res = results

        dense_failed = isinstance(dense_res, Exception)
        sparse_failed = isinstance(sparse_res, Exception)

        if dense_failed and sparse_failed:
            logger.error(
                f"Both retrieval branches failed: dense={dense_res}, lexical={sparse_res}",
                extra={"query_hash": query_fp},
            )
            raise RetrievalError(
                f"Both dense and lexical retrieval branches failed. Dense: {dense_res}; Lexical: {sparse_res}"
            )

        if dense_failed:
            logger.warning(
                f"Dense retrieval branch failed ({dense_res}); continuing with lexical results only.",
                extra={"query_hash": query_fp, "error": str(dense_res)},
            )
            dense_results: list[RetrievalResult] = []
        else:
            dense_results = dense_res  # type: ignore[assignment]

        if sparse_failed:
            logger.warning(
                f"Lexical retrieval branch failed ({sparse_res}); continuing with dense results only.",
                extra={"query_hash": query_fp, "error": str(sparse_res)},
            )
            sparse_results: list[RetrievalResult] = []
        else:
            sparse_results = sparse_res  # type: ignore[assignment]

        # 2. Observability & Overlap telemetry (PII-free)
        dense_ids = {r.chunk_id for r in dense_results}
        sparse_ids = {r.chunk_id for r in sparse_results}
        overlap_ids = dense_ids.intersection(sparse_ids)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"Completed concurrent retrieval in {elapsed_ms:.2f}ms: "
            f"dense={len(dense_results)}, sparse={len(sparse_results)}, overlap={len(overlap_ids)}",
            extra={
                "query_hash": query_fp,
                "query_length": query_len,
                "dense_count": len(dense_results),
                "sparse_count": len(sparse_results),
                "overlap_count": len(overlap_ids),
                "elapsed_ms": round(elapsed_ms, 2),
            },
        )

        # 3. Fuse results and deduplicate
        fused_results = self.fusion_strategy.fuse(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=resolved_final_top_k,
        )

        logger.info(
            f"Hybrid fusion yielded {len(fused_results)} final ranked chunks",
            extra={"final_count": len(fused_results), "query_hash": query_fp},
        )
        return fused_results
