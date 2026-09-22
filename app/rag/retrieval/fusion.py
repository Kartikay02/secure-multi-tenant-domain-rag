"""Fusion strategies for combining multi-modal retrieval results."""

import uuid
from typing import Any

from app.core.logging import get_logger
from app.rag.retrieval.interfaces import FusionStrategyProtocol
from app.rag.vector.domain import RetrievalResult

logger = get_logger("app.rag.retrieval.fusion")


class ReciprocalRankFusion(FusionStrategyProtocol):
    """Reciprocal Rank Fusion (RRF) combining dense and sparse rankings.

    RRF formula:
        RRF_Score(d) = sum_{m in {dense, sparse}} weight_m / (k + rank_m(d))
    where k is a rank constant (default: 60).
    """

    def __init__(
        self,
        k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def fuse(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Combine dense and sparse results using RRF and deduplicate by chunk_id."""
        fused_map: dict[uuid.UUID, dict[str, Any]] = {}

        # 1. Process dense rankings
        for rank, res in enumerate(dense_results, start=1):
            rrf_score = self.dense_weight / (self.k + rank)
            fused_map[res.chunk_id] = {
                "result": res,
                "score": rrf_score,
                "dense_score": res.dense_score if res.dense_score is not None else res.score,
                "sparse_score": None,
                "dense_rank": rank,
                "sparse_rank": None,
            }

        # 2. Process sparse rankings (merge & deduplicate)
        for rank, res in enumerate(sparse_results, start=1):
            rrf_score = self.sparse_weight / (self.k + rank)
            sparse_score_val = res.sparse_score if res.sparse_score is not None else res.score

            if res.chunk_id in fused_map:
                entry = fused_map[res.chunk_id]
                entry["score"] += rrf_score
                entry["sparse_score"] = sparse_score_val
                entry["sparse_rank"] = rank
                # Merge metadata if sparse result contains additional tags
                if res.metadata:
                    entry["result"].metadata.update(res.metadata)
            else:
                fused_map[res.chunk_id] = {
                    "result": res,
                    "score": rrf_score,
                    "dense_score": None,
                    "sparse_score": sparse_score_val,
                    "dense_rank": None,
                    "sparse_rank": rank,
                }

        # 3. Sort by fused score descending
        sorted_entries = sorted(
            fused_map.values(),
            key=lambda item: item["score"],
            reverse=True,
        )

        # 4. Construct unified RetrievalResult instances
        final_results: list[RetrievalResult] = []
        for entry in sorted_entries[:top_k]:
            base = entry["result"]
            meta = dict(base.metadata)
            meta["fusion_method"] = "rrf"
            meta["rrf_dense_rank"] = entry["dense_rank"]
            meta["rrf_sparse_rank"] = entry["sparse_rank"]

            final_results.append(
                RetrievalResult(
                    chunk_id=base.chunk_id,
                    document_id=base.document_id,
                    text=base.text,
                    score=round(entry["score"], 6),
                    dense_score=entry["dense_score"],
                    sparse_score=entry["sparse_score"],
                    metadata=meta,
                )
            )

        logger.debug(
            f"RRF fused {len(dense_results)} dense and {len(sparse_results)} sparse "
            f"candidates into {len(final_results)} top results."
        )
        return final_results


class LinearCombinationFusion(FusionStrategyProtocol):
    """Linear weighted score fusion normalizing disparate score scales into [0.0, 1.0]."""

    def __init__(
        self,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> None:
        total = dense_weight + sparse_weight
        if total <= 0:
            raise ValueError("Sum of dense_weight and sparse_weight must be positive.")
        self.dense_weight = dense_weight / total
        self.sparse_weight = sparse_weight / total

    def fuse(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Combine dense and sparse candidates using normalized linear weighting."""
        fused_map: dict[uuid.UUID, dict[str, Any]] = {}

        for res in dense_results:
            fused_map[res.chunk_id] = {
                "result": res,
                "dense_val": res.score,
                "sparse_val": 0.0,
                "dense_score": res.dense_score if res.dense_score is not None else res.score,
                "sparse_score": res.sparse_score,
            }

        for res in sparse_results:
            s_score_raw = res.sparse_score if res.sparse_score is not None else res.score
            if res.chunk_id in fused_map:
                fused_map[res.chunk_id]["sparse_val"] = res.score
                fused_map[res.chunk_id]["sparse_score"] = s_score_raw
                if res.metadata:
                    fused_map[res.chunk_id]["result"].metadata.update(res.metadata)
            else:
                fused_map[res.chunk_id] = {
                    "result": res,
                    "dense_val": 0.0,
                    "sparse_val": res.score,
                    "dense_score": res.dense_score,
                    "sparse_score": s_score_raw,
                }

        # Calculate weighted score for each candidate
        scored_entries: list[dict[str, Any]] = []
        for entry in fused_map.values():
            d_val = entry["dense_val"]
            s_val = entry["sparse_val"]

            final_score = (self.dense_weight * d_val) + (self.sparse_weight * s_val)
            entry["final_score"] = final_score
            scored_entries.append(entry)

        scored_entries.sort(key=lambda x: x["final_score"], reverse=True)

        final_results: list[RetrievalResult] = []
        for entry in scored_entries[:top_k]:
            base = entry["result"]
            meta = dict(base.metadata)
            meta["fusion_method"] = "linear"

            final_results.append(
                RetrievalResult(
                    chunk_id=base.chunk_id,
                    document_id=base.document_id,
                    text=base.text,
                    score=round(entry["final_score"], 4),
                    dense_score=entry["dense_score"],
                    sparse_score=entry["sparse_score"],
                    metadata=meta,
                )
            )

        return final_results


class RelativeScoreFusion(FusionStrategyProtocol):
    """Normalizes scores relative to max score observed in each branch before weighting."""

    def __init__(
        self,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> None:
        total = dense_weight + sparse_weight
        if total <= 0:
            raise ValueError("Sum of dense_weight and sparse_weight must be positive.")
        self.dense_weight = dense_weight / total
        self.sparse_weight = sparse_weight / total

    def fuse(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Combine candidates after relative max-score normalization."""
        max_dense = max((r.score for r in dense_results), default=1.0) or 1.0
        max_sparse = max((r.score for r in sparse_results), default=1.0) or 1.0

        # Scale dense results
        scaled_dense = [
            RetrievalResult(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                text=r.text,
                score=r.score / max_dense,
                dense_score=r.score,
                sparse_score=r.sparse_score,
                metadata=r.metadata,
            )
            for r in dense_results
        ]

        # Scale sparse results
        scaled_sparse = [
            RetrievalResult(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                text=r.text,
                score=r.score / max_sparse,
                dense_score=r.dense_score,
                sparse_score=r.score,
                metadata=r.metadata,
            )
            for r in sparse_results
        ]

        # Delegate to linear combination with scaled scores
        linear_combiner = LinearCombinationFusion(
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
        )
        fused = linear_combiner.fuse(scaled_dense, scaled_sparse, top_k=top_k)

        # Update metadata tag
        updated: list[RetrievalResult] = []
        for r in fused:
            meta = dict(r.metadata)
            meta["fusion_method"] = "relative_score"
            updated.append(
                RetrievalResult(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    text=r.text,
                    score=r.score,
                    dense_score=r.dense_score,
                    sparse_score=r.sparse_score,
                    metadata=meta,
                )
            )
        return updated
