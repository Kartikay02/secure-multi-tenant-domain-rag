"""LLM-as-a-judge evaluation metrics with explicit categorization and limitation tracking."""

from typing import Any

from app.rag.evaluation.domain import (
    EvaluationPrediction,
    EvaluationSample,
    MetricCategory,
    MetricResult,
)
from app.rag.evaluation.interfaces import EvaluationMetricProtocol, LLMJudgeProtocol


class MockLLMJudge(LLMJudgeProtocol):
    """Deterministic mock LLM judge for testing and offline evaluation without external API calls."""

    def __init__(
        self,
        default_score: float = 0.85,
        default_reasoning: str = "Evaluated via deterministic mock judge.",
    ) -> None:
        self.default_score = default_score
        self.default_reasoning = default_reasoning
        self.call_history: list[dict[str, Any]] = []

    async def judge(
        self,
        prompt: str,
        system_instruction: str | None = None,
    ) -> tuple[float, str, dict[str, Any]]:
        self.call_history.append(
            {
                "prompt": prompt,
                "system_instruction": system_instruction,
            }
        )
        # If the prompt contains a refusal or error marker, adjust score deterministically
        score = self.default_score
        if "ERROR:" in prompt or "EMPTY" in prompt:
            score = 0.0
        elif "PERFECT" in prompt:
            score = 1.0

        return score, self.default_reasoning, {"mock": True, "calls": len(self.call_history)}


class LLMContextRelevanceJudge(EvaluationMetricProtocol):
    """LLM-as-a-judge metric evaluating whether retrieved context documents contain sufficient information to answer the question."""

    def __init__(self, judge_client: LLMJudgeProtocol) -> None:
        self._judge = judge_client

    @property
    def name(self) -> str:
        return "llm_context_relevance"

    @property
    def category(self) -> MetricCategory:
        return MetricCategory.LLM_JUDGE

    @property
    def limitations(self) -> str:
        return (
            "LLM judge for context relevance is subject to position bias (overweighting first/last chunks), "
            "verbosity bias, and stochastic inconsistency across runs."
        )

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

        context_texts = [
            c.get("content", "") if isinstance(c, dict) else str(getattr(c, "content", ""))
            for c in prediction.retrieved_chunks
        ]
        context_body = "\n\n---\n\n".join(context_texts[:10])

        prompt = (
            f"You are an impartial evaluator assessing context relevance for RAG.\n"
            f"Question: {sample.question}\n"
            f"Retrieved Context:\n{context_body}\n\n"
            f"Rate whether the retrieved context contains the information needed to answer the question.\n"
            f"Score between 0.0 (completely irrelevant) and 1.0 (contains complete, relevant evidence)."
        )

        score, reasoning, meta = await self._judge.judge(
            prompt,
            system_instruction="You are an expert AI evaluator scoring RAG context relevance.",
        )

        return MetricResult(
            metric_name=self.name,
            category=self.category,
            score=min(1.0, max(0.0, score)),
            details={"reasoning": reasoning, "judge_metadata": meta},
            limitations=self.limitations,
        )


class LLMAnswerGroundednessJudge(EvaluationMetricProtocol):
    """LLM-as-a-judge metric evaluating factual faithfulness of the answer against retrieved context."""

    def __init__(self, judge_client: LLMJudgeProtocol) -> None:
        self._judge = judge_client

    @property
    def name(self) -> str:
        return "llm_answer_groundedness"

    @property
    def category(self) -> MetricCategory:
        return MetricCategory.LLM_JUDGE

    @property
    def limitations(self) -> str:
        return (
            "LLM judge for groundedness may suffer from self-preference bias, can overlook subtle numerical inaccuracies, "
            "and depends heavily on the reasoning capacity of the judge model."
        )

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

        context_texts = [
            c.get("content", "") if isinstance(c, dict) else str(getattr(c, "content", ""))
            for c in prediction.retrieved_chunks
        ]
        context_body = "\n\n---\n\n".join(context_texts[:10])

        prompt = (
            f"You are an impartial evaluator checking answer faithfulness and hallucinations.\n"
            f"Question: {sample.question}\n"
            f"Context Evidence:\n{context_body}\n\n"
            f"Answer to Evaluate:\n{prediction.answer}\n\n"
            f"Score how faithfully the answer is grounded in the evidence (0.0 to 1.0)."
        )

        score, reasoning, meta = await self._judge.judge(
            prompt,
            system_instruction="You are an expert factual verification judge.",
        )

        return MetricResult(
            metric_name=self.name,
            category=self.category,
            score=min(1.0, max(0.0, score)),
            details={"reasoning": reasoning, "judge_metadata": meta},
            limitations=self.limitations,
        )


class LLMAnswerQualityJudge(EvaluationMetricProtocol):
    """LLM-as-a-judge metric evaluating overall correctness and quality compared to expected answer."""

    def __init__(self, judge_client: LLMJudgeProtocol) -> None:
        self._judge = judge_client

    @property
    def name(self) -> str:
        return "llm_answer_quality"

    @property
    def category(self) -> MetricCategory:
        return MetricCategory.LLM_JUDGE

    @property
    def limitations(self) -> str:
        return (
            "LLM answer quality judge exhibits verbosity bias (favoring longer answers) and may reward stylistically "
            "pleasing text even if slightly ungrounded."
        )

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

        prompt = (
            f"You are an impartial evaluator assessing QA answer quality.\n"
            f"Question: {sample.question}\n"
            f"Expected Reference Answer: {sample.expected_answer or 'N/A'}\n\n"
            f"Candidate Generated Answer:\n{prediction.answer}\n\n"
            f"Score overall answer correctness and completeness (0.0 to 1.0)."
        )

        score, reasoning, meta = await self._judge.judge(
            prompt,
            system_instruction="You are an expert QA evaluation judge.",
        )

        return MetricResult(
            metric_name=self.name,
            category=self.category,
            score=min(1.0, max(0.0, score)),
            details={"reasoning": reasoning, "judge_metadata": meta},
            limitations=self.limitations,
        )
