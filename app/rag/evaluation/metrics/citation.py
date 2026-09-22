"""Automated citation accuracy metrics measuring citation validity and source fidelity."""

import re

from app.rag.evaluation.domain import (
    EvaluationPrediction,
    EvaluationSample,
    MetricCategory,
    MetricResult,
)
from app.rag.evaluation.interfaces import EvaluationMetricProtocol


class CitationAccuracyMetric(EvaluationMetricProtocol):
    """Measures citation validity, presence, and alignment with expected sources."""

    @property
    def name(self) -> str:
        return "citation_accuracy"

    @property
    def category(self) -> MetricCategory:
        return MetricCategory.AUTOMATED

    @property
    def limitations(self) -> str:
        return (
            "Citation accuracy verifies citation index existence and source ID alignment with expected sources. "
            "It cannot verify whether the cited sentence is logically necessary to support the answer."
        )

    def _extract_citation_ids(self, text: str) -> list[int]:
        matches = re.findall(r"\[(\d+)\]", text)
        return [int(m) for m in matches]

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
            # Conservative refusal should not cite sources
            cited = self._extract_citation_ids(prediction.answer)
            score = 1.0 if not cited else 0.0
            return MetricResult(
                metric_name=self.name,
                category=self.category,
                score=score,
                details={"refusal_mode": True, "cited_count": len(cited)},
                limitations=self.limitations,
            )

        cited_ids = self._extract_citation_ids(prediction.answer)
        if not cited_ids and prediction.citations:
            cited_ids = prediction.citations

        # 1. Check validity of cited IDs against context chunks
        total_chunks = len(prediction.retrieved_chunks)
        # Valid citation IDs are typically 1-indexed up to total_chunks (or match chunk metadata citation_id)
        valid_chunk_citation_ids: set[int] = set()
        for idx, chunk in enumerate(prediction.retrieved_chunks, start=1):
            c_id = (
                chunk.get("citation_id")
                if isinstance(chunk, dict)
                else getattr(chunk, "citation_id", idx)
            )
            valid_chunk_citation_ids.add(c_id or idx)

        if not valid_chunk_citation_ids:
            valid_chunk_citation_ids = set(range(1, total_chunks + 1))

        if not cited_ids:
            # No citations present in answer
            score = 1.0 if not sample.expected_sources else 0.0
            return MetricResult(
                metric_name=self.name,
                category=self.category,
                score=score,
                details={
                    "cited_ids": [],
                    "validity_ratio": 0.0,
                    "reason": "No citations found in answer",
                },
                limitations=self.limitations,
            )

        valid_citations = [c for c in cited_ids if c in valid_chunk_citation_ids]
        validity_ratio = len(valid_citations) / len(cited_ids)

        # 2. Check alignment of cited chunks with expected sources
        expected = {s.strip().lower() for s in sample.expected_sources if s.strip()}
        if expected:
            # Map citation ID to source identifier
            cited_sources: set[str] = set()
            for c_id in valid_citations:
                # Find matching chunk
                for idx, chunk in enumerate(prediction.retrieved_chunks, start=1):
                    chunk_c_id = (
                        chunk.get("citation_id", idx)
                        if isinstance(chunk, dict)
                        else getattr(chunk, "citation_id", idx)
                    )
                    if chunk_c_id == c_id:
                        src = (
                            chunk.get("source_id") or chunk.get("document_id") or ""
                            if isinstance(chunk, dict)
                            else str(getattr(chunk, "source_id", getattr(chunk, "document_id", "")))
                        )
                        if src:
                            cited_sources.add(str(src).strip().lower())

            hits = expected.intersection(cited_sources)
            alignment_score = len(hits) / len(expected) if expected else 1.0
            score = (0.5 * validity_ratio) + (0.5 * alignment_score)
        else:
            # When no gold sources are supplied, validity of citation references dominates
            alignment_score = 1.0
            score = validity_ratio

        return MetricResult(
            metric_name=self.name,
            category=self.category,
            score=min(1.0, max(0.0, score)),
            details={
                "cited_ids": cited_ids,
                "valid_citations_count": len(valid_citations),
                "validity_ratio": round(validity_ratio, 3),
                "alignment_score": round(alignment_score, 3),
            },
            limitations=self.limitations,
        )
