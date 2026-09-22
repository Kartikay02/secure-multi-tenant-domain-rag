"""Automated information retrieval metrics: Recall@K, Precision@K, and MRR."""

from app.rag.evaluation.domain import (
    EvaluationPrediction,
    EvaluationSample,
    MetricCategory,
    MetricResult,
)
from app.rag.evaluation.interfaces import EvaluationMetricProtocol


class RecallAtKMetric(EvaluationMetricProtocol):
    """Measures the proportion of expected relevant sources retrieved in top-K candidates."""

    def __init__(self, k: int = 5) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self._k = k

    @property
    def name(self) -> str:
        return f"recall_at_{self._k}"

    @property
    def category(self) -> MetricCategory:
        return MetricCategory.AUTOMATED

    @property
    def limitations(self) -> str:
        return (
            "Recall@K assumes comprehensive ground-truth labeling. If the index contains "
            "unannotated relevant chunks, they do not contribute to recall. Does not penalize "
            "irrelevant retrieved chunks."
        )

    async def evaluate(
        self,
        sample: EvaluationSample,
        prediction: EvaluationPrediction,
    ) -> MetricResult:
        expected = {s.strip().lower() for s in sample.expected_sources if s.strip()}
        if not expected:
            # Edge case: No expected sources specified
            return MetricResult(
                metric_name=self.name,
                category=self.category,
                score=1.0 if not prediction.retrieved_sources else 1.0,
                details={"expected_count": 0, "retrieved_top_k": []},
                limitations=self.limitations,
            )

        retrieved_top_k = [
            s.strip().lower() for s in prediction.retrieved_sources[: self._k] if s.strip()
        ]
        hits = expected.intersection(set(retrieved_top_k))
        score = len(hits) / float(len(expected))

        return MetricResult(
            metric_name=self.name,
            category=self.category,
            score=min(1.0, max(0.0, score)),
            details={
                "k": self._k,
                "expected_count": len(expected),
                "retrieved_count": len(retrieved_top_k),
                "hits_count": len(hits),
                "hits": sorted(list(hits)),
            },
            limitations=self.limitations,
        )


class PrecisionAtKMetric(EvaluationMetricProtocol):
    """Measures the proportion of top-K retrieved sources that are relevant."""

    def __init__(self, k: int = 5) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self._k = k

    @property
    def name(self) -> str:
        return f"precision_at_{self._k}"

    @property
    def category(self) -> MetricCategory:
        return MetricCategory.AUTOMATED

    @property
    def limitations(self) -> str:
        return (
            "Precision@K penalizes valid documents that are absent from gold annotations (false negatives in ground truth). "
            "If fewer than K relevant documents exist in the corpus, maximum achievable precision is strictly < 1.0."
        )

    async def evaluate(
        self,
        sample: EvaluationSample,
        prediction: EvaluationPrediction,
    ) -> MetricResult:
        expected = {s.strip().lower() for s in sample.expected_sources if s.strip()}
        retrieved_top_k = [
            s.strip().lower() for s in prediction.retrieved_sources[: self._k] if s.strip()
        ]

        if not retrieved_top_k:
            return MetricResult(
                metric_name=self.name,
                category=self.category,
                score=1.0 if not expected else 0.0,
                details={"k": self._k, "retrieved_count": 0, "hits_count": 0},
                limitations=self.limitations,
            )

        hits = expected.intersection(set(retrieved_top_k))
        score = len(hits) / float(self._k)

        return MetricResult(
            metric_name=self.name,
            category=self.category,
            score=min(1.0, max(0.0, score)),
            details={
                "k": self._k,
                "expected_count": len(expected),
                "retrieved_count": len(retrieved_top_k),
                "hits_count": len(hits),
                "hits": sorted(list(hits)),
            },
            limitations=self.limitations,
        )


class MRRMetric(EvaluationMetricProtocol):
    """Mean Reciprocal Rank (MRR) evaluating the rank of the first relevant source."""

    def __init__(self, k: int = 10) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self._k = k

    @property
    def name(self) -> str:
        return f"mrr_at_{self._k}"

    @property
    def category(self) -> MetricCategory:
        return MetricCategory.AUTOMATED

    @property
    def limitations(self) -> str:
        return (
            "MRR measures exclusively the rank of the first hit. It ignores whether subsequent chunks "
            "are relevant and provides no measure of multi-chunk evidentiary completeness."
        )

    async def evaluate(
        self,
        sample: EvaluationSample,
        prediction: EvaluationPrediction,
    ) -> MetricResult:
        expected = {s.strip().lower() for s in sample.expected_sources if s.strip()}
        if not expected:
            return MetricResult(
                metric_name=self.name,
                category=self.category,
                score=1.0,
                details={"k": self._k, "first_hit_rank": 0},
                limitations=self.limitations,
            )

        retrieved = [
            s.strip().lower() for s in prediction.retrieved_sources[: self._k] if s.strip()
        ]
        reciprocal_rank = 0.0
        first_hit_rank = 0

        for rank, source_id in enumerate(retrieved, start=1):
            if source_id in expected:
                first_hit_rank = rank
                reciprocal_rank = 1.0 / rank
                break

        return MetricResult(
            metric_name=self.name,
            category=self.category,
            score=reciprocal_rank,
            details={
                "k": self._k,
                "first_hit_rank": first_hit_rank,
                "reciprocal_rank": reciprocal_rank,
            },
            limitations=self.limitations,
        )
