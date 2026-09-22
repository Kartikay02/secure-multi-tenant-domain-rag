"""Automated answer quality metric based on token overlap F1."""

import re
from collections import Counter

from app.rag.evaluation.domain import (
    EvaluationPrediction,
    EvaluationSample,
    MetricCategory,
    MetricResult,
)
from app.rag.evaluation.interfaces import EvaluationMetricProtocol


class AnswerLexicalSimilarityMetric(EvaluationMetricProtocol):
    """Measures lexical token precision, recall, and F1 between generated and expected answer."""

    @property
    def name(self) -> str:
        return "answer_similarity_f1"

    @property
    def category(self) -> MetricCategory:
        return MetricCategory.AUTOMATED

    @property
    def limitations(self) -> str:
        return (
            "Token-level F1 similarity penalizes valid semantic paraphrasing, concise answers, and differing sentence "
            "structures that convey equivalent factual meaning using alternate vocabulary."
        )

    def _normalize_and_tokenize(self, text: str) -> list[str]:
        # Strip citation brackets [1], punctuation, and lower-case
        cleaned = re.sub(r"\[\d+\]", "", text).lower()
        tokens = re.findall(r"\b\w+\b", cleaned)
        return tokens

    async def evaluate(
        self,
        sample: EvaluationSample,
        prediction: EvaluationPrediction,
    ) -> MetricResult:
        if prediction.error:
            return MetricResult(
                metric_name=self.name,
                category=self.category,
                score=0.0,
                details={"reason": f"Prediction error: {prediction.error}"},
                limitations=self.limitations,
            )

        if not sample.expected_answer:
            return MetricResult(
                metric_name=self.name,
                category=self.category,
                score=1.0,
                details={"reason": "No expected_answer provided for sample"},
                limitations=self.limitations,
            )

        pred_tokens = self._normalize_and_tokenize(prediction.answer)
        target_tokens = self._normalize_and_tokenize(sample.expected_answer)

        if not pred_tokens or not target_tokens:
            score = 1.0 if pred_tokens == target_tokens else 0.0
            return MetricResult(
                metric_name=self.name,
                category=self.category,
                score=score,
                details={
                    "pred_token_count": len(pred_tokens),
                    "target_token_count": len(target_tokens),
                },
                limitations=self.limitations,
            )

        common = Counter(pred_tokens) & Counter(target_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return MetricResult(
                metric_name=self.name,
                category=self.category,
                score=0.0,
                details={"precision": 0.0, "recall": 0.0, "f1": 0.0},
                limitations=self.limitations,
            )

        precision = num_same / len(pred_tokens)
        recall = num_same / len(target_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        return MetricResult(
            metric_name=self.name,
            category=self.category,
            score=round(f1, 4),
            details={
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "overlapping_tokens": num_same,
            },
            limitations=self.limitations,
        )
