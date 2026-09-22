"""Unit tests for automated RAG evaluation metrics."""

import pytest

from app.rag.evaluation.domain import (
    EvaluationPrediction,
    EvaluationSample,
    MetricCategory,
)
from app.rag.evaluation.metrics.answer_quality import AnswerLexicalSimilarityMetric
from app.rag.evaluation.metrics.citation import CitationAccuracyMetric
from app.rag.evaluation.metrics.groundedness import AnswerGroundednessMetric
from app.rag.evaluation.metrics.relevance import ContextRelevanceMetric
from app.rag.evaluation.metrics.retrieval import (
    MRRMetric,
    PrecisionAtKMetric,
    RecallAtKMetric,
)


@pytest.mark.asyncio
async def test_recall_at_k_exact_and_partial() -> None:
    metric = RecallAtKMetric(k=3)
    assert metric.category == MetricCategory.AUTOMATED
    assert "Recall@K assumes" in metric.limitations

    sample = EvaluationSample(
        question="Query",
        expected_sources=["doc_1", "doc_2", "doc_3", "doc_4"],
    )

    # 1. 2 of 4 found in top 3
    pred_partial = EvaluationPrediction(
        sample_id="1",
        question="Query",
        retrieved_sources=["doc_1", "doc_2", "doc_other", "doc_3"],
    )
    res_partial = await metric.evaluate(sample, pred_partial)
    assert res_partial.score == 0.5  # 2 / 4
    assert res_partial.details["hits_count"] == 2

    # 2. Complete recall in top 3
    sample_small = EvaluationSample(
        question="Query",
        expected_sources=["doc_1", "doc_2"],
    )
    res_full = await metric.evaluate(sample_small, pred_partial)
    assert res_full.score == 1.0  # 2 / 2

    # 3. Zero hits
    pred_none = EvaluationPrediction(
        sample_id="1",
        question="Query",
        retrieved_sources=["unrelated_1", "unrelated_2"],
    )
    res_none = await metric.evaluate(sample_small, pred_none)
    assert res_none.score == 0.0


@pytest.mark.asyncio
async def test_precision_at_k() -> None:
    metric = PrecisionAtKMetric(k=4)
    assert metric.category == MetricCategory.AUTOMATED

    sample = EvaluationSample(
        question="Query",
        expected_sources=["doc_1", "doc_2"],
    )

    # 2 out of top 4 are expected -> precision = 2 / 4 = 0.5
    pred = EvaluationPrediction(
        sample_id="1",
        question="Query",
        retrieved_sources=["doc_1", "unrelated", "doc_2", "unrelated_2"],
    )
    res = await metric.evaluate(sample, pred)
    assert res.score == 0.5

    # 0 out of top 4 -> 0.0
    pred_zero = EvaluationPrediction(
        sample_id="1",
        question="Query",
        retrieved_sources=["unrelated_1", "unrelated_2", "unrelated_3", "unrelated_4"],
    )
    res_zero = await metric.evaluate(sample, pred_zero)
    assert res_zero.score == 0.0


@pytest.mark.asyncio
async def test_mrr_metric() -> None:
    metric = MRRMetric(k=5)
    assert metric.category == MetricCategory.AUTOMATED

    sample = EvaluationSample(
        question="Query",
        expected_sources=["gold_doc"],
    )

    # First hit at rank 1 -> MRR = 1.0
    pred_rank1 = EvaluationPrediction(
        sample_id="1",
        question="Query",
        retrieved_sources=["gold_doc", "doc_2", "doc_3"],
    )
    res1 = await metric.evaluate(sample, pred_rank1)
    assert res1.score == 1.0
    assert res1.details["first_hit_rank"] == 1

    # First hit at rank 2 -> MRR = 0.5
    pred_rank2 = EvaluationPrediction(
        sample_id="1",
        question="Query",
        retrieved_sources=["other_doc", "gold_doc", "doc_3"],
    )
    res2 = await metric.evaluate(sample, pred_rank2)
    assert res2.score == 0.5
    assert res2.details["first_hit_rank"] == 2

    # First hit at rank 4 -> MRR = 0.25
    pred_rank4 = EvaluationPrediction(
        sample_id="1",
        question="Query",
        retrieved_sources=["other_1", "other_2", "other_3", "gold_doc", "other_5"],
    )
    res4 = await metric.evaluate(sample, pred_rank4)
    assert res4.score == 0.25

    # No hits in top-K -> MRR = 0.0
    pred_none = EvaluationPrediction(
        sample_id="1",
        question="Query",
        retrieved_sources=["other_1", "other_2"],
    )
    res_none = await metric.evaluate(sample, pred_none)
    assert res_none.score == 0.0


@pytest.mark.asyncio
async def test_context_relevance_metric() -> None:
    metric = ContextRelevanceMetric()
    assert metric.category == MetricCategory.AUTOMATED

    sample = EvaluationSample(
        question="What are HNSW index parameters for vector search?",
    )

    # Chunks containing the question terms
    pred_relevant = EvaluationPrediction(
        sample_id="1",
        question=sample.question,
        retrieved_chunks=[
            {"content": "HNSW index vector search uses ef_construction and M parameters."},
            {
                "content": "Parameters determine the graph connectivity for nearest neighbors search."
            },
        ],
    )
    res_rel = await metric.evaluate(sample, pred_relevant)
    assert res_rel.score > 0.7
    assert res_rel.details["covered_terms_count"] >= 3

    # Completely irrelevant chunks
    pred_irrelevant = EvaluationPrediction(
        sample_id="1",
        question=sample.question,
        retrieved_chunks=[
            {
                "content": "Baking sourdough bread requires flour, water, salt, and wild yeast culture."
            },
        ],
    )
    res_irrel = await metric.evaluate(sample, pred_irrelevant)
    assert res_irrel.score == 0.0


@pytest.mark.asyncio
async def test_answer_groundedness_metric() -> None:
    metric = AnswerGroundednessMetric()

    sample = EvaluationSample(
        question="How does hybrid retrieval rank documents?",
        expected_sources=["doc-01"],
    )

    # Grounded answer: claims supported by context chunks
    pred_grounded = EvaluationPrediction(
        sample_id="1",
        question=sample.question,
        answer="Hybrid retrieval combines dense and lexical rankings using Reciprocal Rank Fusion [1].",
        retrieved_chunks=[
            {
                "content": "Hybrid retrieval combines dense and lexical rankings using Reciprocal Rank Fusion."
            }
        ],
        grounded=True,
    )
    res_grounded = await metric.evaluate(sample, pred_grounded)
    assert res_grounded.score >= 0.8
    assert res_grounded.details["supported_claims"] >= 1

    # Ungrounded / hallucinated answer
    pred_hallucinated = EvaluationPrediction(
        sample_id="1",
        question=sample.question,
        answer="Documents are teleported to a quantum computing cluster via alien warp drives [1].",
        retrieved_chunks=[
            {
                "content": "Hybrid retrieval combines dense and lexical rankings using Reciprocal Rank Fusion."
            }
        ],
        grounded=False,
    )
    res_hallucinated = await metric.evaluate(sample, pred_hallucinated)
    assert res_hallucinated.score < 0.3


@pytest.mark.asyncio
async def test_citation_accuracy_metric() -> None:
    metric = CitationAccuracyMetric()

    sample = EvaluationSample(
        question="Explain indexing",
        expected_sources=["doc-01"],
    )

    # Valid citation [1] matching chunk citation_id 1 and source doc-01
    pred_valid = EvaluationPrediction(
        sample_id="1",
        question=sample.question,
        answer="Indexing organizes tokens into inverted lists [1].",
        retrieved_chunks=[
            {"citation_id": 1, "source_id": "doc-01", "content": "Indexing organizes tokens."}
        ],
    )
    res_valid = await metric.evaluate(sample, pred_valid)
    assert res_valid.score == 1.0
    assert res_valid.details["validity_ratio"] == 1.0

    # Invalid citation [99] which doesn't exist in chunks
    pred_invalid = EvaluationPrediction(
        sample_id="1",
        question=sample.question,
        answer="Indexing organizes tokens into inverted lists [99].",
        retrieved_chunks=[
            {"citation_id": 1, "source_id": "doc-01", "content": "Indexing organizes tokens."}
        ],
    )
    res_invalid = await metric.evaluate(sample, pred_invalid)
    assert res_invalid.score == 0.0


@pytest.mark.asyncio
async def test_answer_lexical_similarity_f1() -> None:
    metric = AnswerLexicalSimilarityMetric()

    sample = EvaluationSample(
        question="What is the speed of light?",
        expected_answer="The speed of light in vacuum is approximately 300,000 kilometers per second.",
    )

    # Exact match (modulo citations) -> F1 = 1.0
    pred_exact = EvaluationPrediction(
        sample_id="1",
        question=sample.question,
        answer="The speed of light in vacuum is approximately 300,000 kilometers per second [1].",
    )
    res_exact = await metric.evaluate(sample, pred_exact)
    assert res_exact.score == 1.0

    # Partial match
    pred_partial = EvaluationPrediction(
        sample_id="1",
        question=sample.question,
        answer="The speed of light is 300,000 kilometers.",
    )
    res_partial = await metric.evaluate(sample, pred_partial)
    assert 0.4 < res_partial.score < 1.0
    assert res_partial.details["f1"] == res_partial.score

    # Disjoint text -> F1 = 0.0
    pred_disjoint = EvaluationPrediction(
        sample_id="1",
        question=sample.question,
        answer="Photosynthesis requires chlorophyll and sunlight.",
    )
    res_disjoint = await metric.evaluate(sample, pred_disjoint)
    assert res_disjoint.score == 0.0
