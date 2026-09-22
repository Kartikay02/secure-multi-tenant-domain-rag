"""Automated context relevance metrics measuring query-to-context alignment."""

import re

from app.rag.evaluation.domain import (
    EvaluationPrediction,
    EvaluationSample,
    MetricCategory,
    MetricResult,
)
from app.rag.evaluation.interfaces import EvaluationMetricProtocol

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "can't",
    "cannot",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


class ContextRelevanceMetric(EvaluationMetricProtocol):
    """Automated metric evaluating query token coverage and relevance across retrieved chunks."""

    @property
    def name(self) -> str:
        return "context_relevance"

    @property
    def category(self) -> MetricCategory:
        return MetricCategory.AUTOMATED

    @property
    def limitations(self) -> str:
        return (
            "Lexical context relevance evaluates keyword presence and does not evaluate deeper semantic "
            "intent or deductive relevance. May reward chunks that contain query keywords out of context."
        )

    def _extract_keywords(self, text: str) -> set[str]:
        words = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower())
        return {w for w in words if w not in STOPWORDS}

    async def evaluate(
        self,
        sample: EvaluationSample,
        prediction: EvaluationPrediction,
    ) -> MetricResult:
        query_terms = self._extract_keywords(sample.question)
        if not query_terms:
            return MetricResult(
                metric_name=self.name,
                category=self.category,
                score=1.0,
                details={"query_terms_count": 0, "covered_terms_count": 0},
                limitations=self.limitations,
            )

        # Collect text from chunks
        chunk_texts: list[str] = []
        for chunk in prediction.retrieved_chunks:
            if isinstance(chunk, dict):
                chunk_texts.append(chunk.get("content", "") or chunk.get("text", ""))
            elif hasattr(chunk, "content"):
                chunk_texts.append(str(chunk.content))

        if not chunk_texts:
            return MetricResult(
                metric_name=self.name,
                category=self.category,
                score=0.0,
                details={"query_terms_count": len(query_terms), "chunks_count": 0},
                limitations=self.limitations,
            )

        combined_context = " ".join(chunk_texts).lower()
        covered_terms = {t for t in query_terms if t in combined_context}
        coverage_ratio = len(covered_terms) / len(query_terms)

        # Fraction of chunks containing at least one query term
        relevant_chunks_count = 0
        for text in chunk_texts:
            text_lower = text.lower()
            if any(t in text_lower for t in query_terms):
                relevant_chunks_count += 1
        chunk_density = relevant_chunks_count / len(chunk_texts)

        # Weighted aggregate: 70% query term coverage, 30% chunk density
        score = (0.7 * coverage_ratio) + (0.3 * chunk_density)

        return MetricResult(
            metric_name=self.name,
            category=self.category,
            score=min(1.0, max(0.0, score)),
            details={
                "query_terms_count": len(query_terms),
                "covered_terms_count": len(covered_terms),
                "coverage_ratio": round(coverage_ratio, 4),
                "chunk_density": round(chunk_density, 4),
                "chunks_evaluated": len(chunk_texts),
            },
            limitations=self.limitations,
        )
