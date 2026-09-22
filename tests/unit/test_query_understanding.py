"""Unit tests for QueryClassifier, QueryExpander, and QueryRouter."""

import pytest

from app.rag.query.classifier import QueryClassifier, QueryIntent
from app.rag.query.expander import QueryExpander
from app.rag.query.router import QueryRouter


class TestQueryClassifier:
    """Validate zero-latency regex intent classification across all semantic intents."""

    @pytest.fixture
    def classifier(self) -> QueryClassifier:
        return QueryClassifier()

    def test_classify_summary_intent(self, classifier: QueryClassifier) -> None:
        queries = [
            "Can you summarize the architecture?",
            "Give me a high-level overview of this system",
            "What is this document about?",
            "tl;dr of the document",
            "What are the main points in this report?",
        ]
        for q in queries:
            result = classifier.classify(q)
            assert result.intent == QueryIntent.SUMMARY
            assert result.confidence >= 0.90
            assert result.matched_pattern is not None

    def test_classify_comparison_intent(self, classifier: QueryClassifier) -> None:
        queries = [
            "What is the difference between dense and lexical retrieval?",
            "Compare HNSW versus IVFFlat indexing",
            "What are the trade-offs of using cross-encoders?",
            "BM25 vs vector search pros and cons",
        ]
        for q in queries:
            result = classifier.classify(q)
            assert result.intent == QueryIntent.COMPARISON
            assert result.confidence >= 0.85

    def test_classify_definition_intent(self, classifier: QueryClassifier) -> None:
        queries = [
            "What is Reciprocal Rank Fusion?",
            "Define semantic chunking",
            "What does HNSW mean in vector databases?",
        ]
        for q in queries:
            result = classifier.classify(q)
            assert result.intent == QueryIntent.DEFINITION
            assert result.confidence >= 0.85

    def test_classify_multi_hop_intent(self, classifier: QueryClassifier) -> None:
        queries = [
            "Which parser is used for PDF and how does chunking split pages?",
            "After ingestion completes what happens next and how does retrieval work?",
        ]
        for q in queries:
            result = classifier.classify(q)
            assert result.intent == QueryIntent.MULTI_HOP
            assert result.confidence >= 0.80

    def test_classify_factual_intent(self, classifier: QueryClassifier) -> None:
        queries = [
            "How many chunks are produced by default?",
            "What is the default value of rerank_top_k?",
            "What parameters control HNSW index construction?",
            "Where is the configuration file loaded from?",
        ]
        for q in queries:
            result = classifier.classify(q)
            assert result.intent == QueryIntent.FACTUAL
            assert result.confidence >= 0.80

    def test_classify_exploratory_fallback(self, classifier: QueryClassifier) -> None:
        queries = [
            "Tell me about scaling considerations",
            "Best practices for production deployments",
        ]
        for q in queries:
            result = classifier.classify(q)
            assert result.intent == QueryIntent.EXPLORATORY
            assert result.confidence >= 0.70

    def test_classify_empty_query(self, classifier: QueryClassifier) -> None:
        result = classifier.classify("   ")
        assert result.intent == QueryIntent.EXPLORATORY
        assert result.confidence == 0.5


class TestQueryExpander:
    """Validate keyword extraction, acronym expansion, and query decomposition."""

    @pytest.fixture
    def expander(self) -> QueryExpander:
        return QueryExpander()

    def test_extract_keywords_removes_stopwords(self, expander: QueryExpander) -> None:
        query = "What is the exact default score threshold for vector retrieval?"
        keywords = expander.extract_keywords(query)
        assert "default" in keywords
        assert "score" in keywords
        assert "threshold" in keywords
        assert "vector" in keywords
        assert "retrieval" in keywords
        assert "is" not in keywords
        assert "the" not in keywords
        assert "for" not in keywords

    def test_expand_for_lexical_acronyms(self, expander: QueryExpander) -> None:
        query = "Explain RAG with RRF and HNSW"
        expanded = expander.expand_for_lexical(query)
        assert (
            "retrieval-augmented generation" in expanded
            or "retrieval augmented generation" in expanded
        )
        assert "reciprocal rank fusion" in expanded
        assert "hierarchical navigable small world" in expanded

    def test_decompose_compound_query(self, expander: QueryExpander) -> None:
        query = "How does dense retrieval work and how does reranking reorder candidates?"
        parts = expander.decompose_query(query)
        assert len(parts) >= 2
        assert query in parts
        assert any("reranking" in p.lower() for p in parts)


class TestQueryRouter:
    """Validate dynamic routing and parameter selection based on semantic intent."""

    @pytest.fixture
    def router(self) -> QueryRouter:
        return QueryRouter(
            default_retrieval_top_k=10,
            default_rerank_top_k=5,
            default_score_threshold=0.65,
        )

    def test_route_summary_intent_boosts_candidates(self, router: QueryRouter) -> None:
        config = router.route(QueryIntent.SUMMARY)
        assert config.retrieval_top_k >= 12
        assert config.rerank_top_k >= 6
        assert config.boost_chunk_zero is True
        assert config.enable_reranking is True

    def test_route_factual_intent_precision(self, router: QueryRouter) -> None:
        config = router.route(QueryIntent.FACTUAL)
        assert config.retrieval_top_k == 10
        assert config.rerank_top_k == 5
        assert config.boost_chunk_zero is False
        assert config.dense_weight >= 0.55

    def test_route_definition_intent_boosts_lexical(self, router: QueryRouter) -> None:
        config = router.route(QueryIntent.DEFINITION)
        assert config.lexical_weight > config.dense_weight

    def test_route_overrides_respected(self, router: QueryRouter) -> None:
        config = router.route(
            QueryIntent.SUMMARY,
            override_top_k=25,
            override_rerank_k=10,
            override_score_threshold=0.8,
        )
        assert config.retrieval_top_k == 25
        assert config.rerank_top_k == 10
        assert config.score_threshold == 0.8
