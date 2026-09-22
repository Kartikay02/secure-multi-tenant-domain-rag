"""Dynamic query router translating query intent into retrieval and ranking configurations."""

from dataclasses import dataclass

from app.rag.query.classifier import QueryIntent


@dataclass(frozen=True)
class QueryRouteConfig:
    """Configured execution parameters for a given query intent."""

    intent: QueryIntent
    retrieval_top_k: int
    rerank_top_k: int
    score_threshold: float | None
    boost_chunk_zero: bool
    enable_reranking: bool
    dense_weight: float
    lexical_weight: float


class QueryRouter:
    """Selects and tailors pipeline parameters based on semantic query intent."""

    def __init__(
        self,
        default_retrieval_top_k: int = 10,
        default_rerank_top_k: int = 5,
        default_score_threshold: float | None = None,
    ) -> None:
        self.default_retrieval_top_k = default_retrieval_top_k
        self.default_rerank_top_k = default_rerank_top_k
        self.default_score_threshold = default_score_threshold

    def route(
        self,
        intent: QueryIntent,
        override_top_k: int | None = None,
        override_rerank_k: int | None = None,
        override_score_threshold: float | None = None,
    ) -> QueryRouteConfig:
        """Derive optimal pipeline parameters for the classified query intent."""
        # 1. Resolve Retrieval Top-K (caller override takes strict precedence)
        if override_top_k is not None:
            resolved_retrieval_k = override_top_k
        else:
            match intent:
                case QueryIntent.SUMMARY:
                    resolved_retrieval_k = max(self.default_retrieval_top_k, 12)
                case QueryIntent.COMPARISON:
                    resolved_retrieval_k = max(self.default_retrieval_top_k, 14)
                case QueryIntent.MULTI_HOP:
                    resolved_retrieval_k = max(self.default_retrieval_top_k, 15)
                case _:
                    resolved_retrieval_k = self.default_retrieval_top_k

        # 2. Resolve Rerank Top-K
        if override_rerank_k is not None:
            resolved_rerank_k = override_rerank_k
        else:
            match intent:
                case QueryIntent.SUMMARY | QueryIntent.COMPARISON:
                    resolved_rerank_k = max(self.default_rerank_top_k, 6)
                case QueryIntent.MULTI_HOP:
                    resolved_rerank_k = max(self.default_rerank_top_k, 7)
                case QueryIntent.DEFINITION:
                    resolved_rerank_k = min(self.default_rerank_top_k, 4)
                case _:
                    resolved_rerank_k = self.default_rerank_top_k

        # 3. Resolve Score Threshold
        resolved_threshold: float | None
        if override_score_threshold is not None:
            resolved_threshold = override_score_threshold
        elif intent == QueryIntent.SUMMARY:
            resolved_threshold = (
                0.0
                if self.default_score_threshold is None
                else max(0.0, self.default_score_threshold - 0.1)
            )
        else:
            resolved_threshold = self.default_score_threshold

        # 4. Intent-specific weights and chunk boosting
        boost_chunk_zero = intent == QueryIntent.SUMMARY
        dense_weight = 0.5
        lexical_weight = 0.5

        if intent == QueryIntent.FACTUAL:
            dense_weight = 0.6
            lexical_weight = 0.4
        elif intent == QueryIntent.DEFINITION:
            dense_weight = 0.4
            lexical_weight = 0.6
        elif intent == QueryIntent.MULTI_HOP:
            dense_weight = 0.55
            lexical_weight = 0.45

        return QueryRouteConfig(
            intent=intent,
            retrieval_top_k=resolved_retrieval_k,
            rerank_top_k=resolved_rerank_k,
            score_threshold=resolved_threshold,
            boost_chunk_zero=boost_chunk_zero,
            enable_reranking=True,
            dense_weight=dense_weight,
            lexical_weight=lexical_weight,
        )
