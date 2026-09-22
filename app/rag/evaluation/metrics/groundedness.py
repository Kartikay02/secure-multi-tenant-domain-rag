"""Automated answer groundedness metric measuring evidentiary fidelity."""

import re

from app.rag.evaluation.domain import (
    EvaluationPrediction,
    EvaluationSample,
    MetricCategory,
    MetricResult,
)
from app.rag.evaluation.interfaces import EvaluationMetricProtocol
from app.rag.evaluation.metrics.relevance import STOPWORDS


class AnswerGroundednessMetric(EvaluationMetricProtocol):
    """Automated metric evaluating whether claims in the generated answer are grounded in retrieved context."""

    @property
    def name(self) -> str:
        return "answer_groundedness"

    @property
    def category(self) -> MetricCategory:
        return MetricCategory.AUTOMATED

    @property
    def limitations(self) -> str:
        return (
            "Automated groundedness relies on token overlap, phrase preservation, and guardrail verdicts. "
            "It cannot execute deep Natural Language Inference (NLI) to identify logical contradictions or fallacies."
        )

    def _split_into_claims(self, text: str) -> list[str]:
        # Strip citation markers
        clean_text = re.sub(r"\[\d+\]", "", text).strip()
        if not clean_text:
            return []
        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", clean_text)
        return [s.strip() for s in sentences if len(s.strip()) > 5]

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

        if prediction.insufficient_context or prediction.fallback_applied:
            # Conservative refusal response
            # A correct refusal when no evidence exists is a grounded response (1.0)
            return MetricResult(
                metric_name=self.name,
                category=self.category,
                score=1.0 if not sample.expected_sources else 0.5,
                details={
                    "fallback_applied": prediction.fallback_applied,
                    "insufficient_context": prediction.insufficient_context,
                    "reason": "Conservative refusal or insufficient context flagged",
                },
                limitations=self.limitations,
            )

        answer_text = prediction.answer.strip()
        if not answer_text:
            return MetricResult(
                metric_name=self.name,
                category=self.category,
                score=0.0,
                details={"reason": "Empty answer text"},
                limitations=self.limitations,
            )

        # Collect context text
        chunk_texts: list[str] = []
        for chunk in prediction.retrieved_chunks:
            if isinstance(chunk, dict):
                chunk_texts.append(chunk.get("content", "") or chunk.get("text", ""))
            elif hasattr(chunk, "content"):
                chunk_texts.append(str(chunk.content))
        context_body = " ".join(chunk_texts).lower()

        claims = self._split_into_claims(answer_text)
        if not claims:
            return MetricResult(
                metric_name=self.name,
                category=self.category,
                score=1.0 if prediction.grounded else 0.5,
                details={"claims_count": 0},
                limitations=self.limitations,
            )

        supported_claims = 0
        claim_details: list[dict[str, float | str | bool]] = []

        for claim in claims:
            words = [
                w.lower()
                for w in re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", claim)
                if w.lower() not in STOPWORDS
            ]
            if not words:
                supported_claims += 1
                claim_details.append({"claim": claim, "supported": True, "overlap_ratio": 1.0})
                continue

            matches = sum(1 for w in words if w in context_body)
            overlap_ratio = matches / len(words)
            is_supported = overlap_ratio >= 0.55

            if is_supported:
                supported_claims += 1
            claim_details.append(
                {
                    "claim": claim[:80],
                    "supported": is_supported,
                    "overlap_ratio": round(overlap_ratio, 3),
                }
            )

        claim_support_ratio = supported_claims / len(claims)

        # If prediction pipeline flagged grounded=False, bound the score
        if not prediction.grounded:
            score = min(0.4, claim_support_ratio * 0.5)
        else:
            score = claim_support_ratio

        return MetricResult(
            metric_name=self.name,
            category=self.category,
            score=min(1.0, max(0.0, score)),
            details={
                "total_claims": len(claims),
                "supported_claims": supported_claims,
                "claim_support_ratio": round(claim_support_ratio, 3),
                "pipeline_grounded_flag": prediction.grounded,
                "claim_details": claim_details,
            },
            limitations=self.limitations,
        )
