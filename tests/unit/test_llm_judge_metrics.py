"""Unit tests for LLM-as-a-judge evaluation metrics."""

import pytest

from app.rag.evaluation.domain import (
    EvaluationPrediction,
    EvaluationSample,
    MetricCategory,
)
from app.rag.evaluation.metrics.llm_judge import (
    LLMAnswerGroundednessJudge,
    LLMAnswerQualityJudge,
    LLMContextRelevanceJudge,
    MockLLMJudge,
)


@pytest.mark.asyncio
async def test_mock_llm_judge() -> None:
    judge = MockLLMJudge(default_score=0.9, default_reasoning="Mock evaluation passes.")
    score, reason, meta = await judge.judge("Evaluate prompt")
    assert score == 0.9
    assert reason == "Mock evaluation passes."
    assert meta["mock"] is True
    assert len(judge.call_history) == 1

    # Prompt with error keyword triggers 0.0
    score_err, _, _ = await judge.judge("ERROR: something broke")
    assert score_err == 0.0


@pytest.mark.asyncio
async def test_llm_judge_metrics_distinction_and_limitations() -> None:
    judge = MockLLMJudge(default_score=0.88)

    relevance_judge = LLMContextRelevanceJudge(judge)
    groundedness_judge = LLMAnswerGroundednessJudge(judge)
    quality_judge = LLMAnswerQualityJudge(judge)

    # 1. Distinct category separation
    assert relevance_judge.category == MetricCategory.LLM_JUDGE
    assert groundedness_judge.category == MetricCategory.LLM_JUDGE
    assert quality_judge.category == MetricCategory.LLM_JUDGE

    # 2. Documented limitations
    assert "position bias" in relevance_judge.limitations
    assert "self-preference" in groundedness_judge.limitations
    assert "verbosity bias" in quality_judge.limitations

    # 3. Execution
    sample = EvaluationSample(
        question="What is pgvector?",
        expected_answer="pgvector is an open-source vector similarity search extension for PostgreSQL.",
    )
    prediction = EvaluationPrediction(
        sample_id=sample.sample_id,
        question=sample.question,
        answer="pgvector provides vector search for PostgreSQL [1].",
        retrieved_chunks=[
            {"content": "pgvector is a vector similarity search extension for PostgreSQL."}
        ],
    )

    res_rel = await relevance_judge.evaluate(sample, prediction)
    assert res_rel.score == 0.88
    assert res_rel.category == MetricCategory.LLM_JUDGE

    res_ground = await groundedness_judge.evaluate(sample, prediction)
    assert res_ground.score == 0.88
    assert res_ground.category == MetricCategory.LLM_JUDGE

    res_qual = await quality_judge.evaluate(sample, prediction)
    assert res_qual.score == 0.88
    assert res_qual.category == MetricCategory.LLM_JUDGE
