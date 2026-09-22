"""End-to-end integration test for the Phase 14 RAG evaluation framework."""

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.rag.context.domain import ContextDocument
from app.rag.evaluation.comparator import RunComparator
from app.rag.evaluation.domain import (
    EvaluationDataset,
    EvaluationSample,
    MetricCategory,
)
from app.rag.evaluation.metrics.answer_quality import AnswerLexicalSimilarityMetric
from app.rag.evaluation.metrics.citation import CitationAccuracyMetric
from app.rag.evaluation.metrics.groundedness import AnswerGroundednessMetric
from app.rag.evaluation.metrics.llm_judge import (
    LLMAnswerGroundednessJudge,
    LLMAnswerQualityJudge,
    LLMContextRelevanceJudge,
    MockLLMJudge,
)
from app.rag.evaluation.metrics.relevance import ContextRelevanceMetric
from app.rag.evaluation.metrics.retrieval import (
    MRRMetric,
    PrecisionAtKMetric,
    RecallAtKMetric,
)
from app.rag.evaluation.reports import ReportGenerator
from app.rag.evaluation.runner import EvaluationRunner, RAGOrchestratorTarget
from app.rag.evaluation.store import FileEvaluationStore
from app.rag.orchestration.domain import PipelineStageLatency, RAGResponse


@pytest.mark.asyncio
async def test_full_rag_evaluation_lifecycle(tmp_path: Path) -> None:
    """Execute complete end-to-end benchmark evaluation lifecycle:

    1. Dataset construction
    2. Multi-metric suite (Automated + LLM Judge)
    3. Target execution via RAGOrchestratorTarget
    4. Batch execution & statistical aggregation
    5. Run persistence & retrieval
    6. Candidate run comparison & delta computation
    7. Markdown summary report generation
    """
    # 1. Dataset construction
    dataset = EvaluationDataset(
        name="domain_rag_prod_bench",
        version="1.0.0",
        description="Production QA evaluation dataset for Domain RAG pipeline",
        samples=[
            EvaluationSample(
                sample_id="eval-001",
                question="How does hybrid retrieval combine dense and lexical search?",
                expected_answer="Hybrid retrieval combines dense vector search with lexical BM25 rankings via Reciprocal Rank Fusion.",
                expected_sources=["retrieval_manual.md"],
                metadata={"category": "architecture", "difficulty": "easy"},
            ),
            EvaluationSample(
                sample_id="eval-002",
                question="What parameters control HNSW index construction?",
                expected_answer="HNSW index construction is governed by ef_construction and M parameters.",
                expected_sources=["vector_config.md"],
                metadata={"category": "indexing", "difficulty": "medium"},
            ),
            EvaluationSample(
                sample_id="eval-003",
                question="How does cross-encoder reranking calibrate scores?",
                expected_answer="Cross-encoder reranking recalibrates relevance scores using deep token-level cross-attention before context assembly.",
                expected_sources=["reranker_guide.md"],
                metadata={"category": "reranking", "difficulty": "hard"},
            ),
        ],
    )

    # 2. Configure metric suite
    mock_judge = MockLLMJudge(default_score=0.90)
    metrics = [
        RecallAtKMetric(k=3),
        PrecisionAtKMetric(k=3),
        MRRMetric(k=5),
        ContextRelevanceMetric(),
        AnswerGroundednessMetric(),
        CitationAccuracyMetric(),
        AnswerLexicalSimilarityMetric(),
        LLMContextRelevanceJudge(mock_judge),
        LLMAnswerGroundednessJudge(mock_judge),
        LLMAnswerQualityJudge(mock_judge),
    ]

    # Verify category distinction
    automated_count = sum(1 for m in metrics if m.category == MetricCategory.AUTOMATED)
    judge_count = sum(1 for m in metrics if m.category == MetricCategory.LLM_JUDGE)
    assert automated_count == 7
    assert judge_count == 3

    # 3. Wire RAG Orchestrator Target (Baseline Pipeline)
    doc1_id, chunk1_id = uuid.uuid4(), uuid.uuid4()
    doc2_id, chunk2_id = uuid.uuid4(), uuid.uuid4()
    doc3_id, chunk3_id = uuid.uuid4(), uuid.uuid4()

    doc_map = {
        "eval-001": ContextDocument(
            citation_id=1,
            citation_label="[1]",
            chunk_id=chunk1_id,
            document_id=doc1_id,
            source_id="retrieval_manual.md",
            title="Retrieval Manual",
            page_number=1,
            score=0.95,
            content="Hybrid retrieval combines dense vector search with lexical BM25 rankings via Reciprocal Rank Fusion.",
            token_count=20,
        ),
        "eval-002": ContextDocument(
            citation_id=1,
            citation_label="[1]",
            chunk_id=chunk2_id,
            document_id=doc2_id,
            source_id="vector_config.md",
            title="Vector Config",
            page_number=3,
            score=0.91,
            content="HNSW index construction is governed by ef_construction and M parameters for graph connectivity.",
            token_count=18,
        ),
        "eval-003": ContextDocument(
            citation_id=1,
            citation_label="[1]",
            chunk_id=chunk3_id,
            document_id=doc3_id,
            source_id="reranker_guide.md",
            title="Reranker Guide",
            page_number=2,
            score=0.88,
            content="Cross-encoder reranking recalibrates relevance scores using deep token-level cross-attention before context assembly.",
            token_count=22,
        ),
    }

    async def mock_execute_baseline(
        query: str, request_id: str | None = None, **kwargs
    ) -> RAGResponse:
        s_id = request_id.replace("eval-", "") if request_id else "eval-001"
        # Find matching sample
        sample = next((s for s in dataset.samples if s_id in s.sample_id), dataset.samples[0])
        doc = doc_map[sample.sample_id]

        return RAGResponse(
            query=query,
            raw_query=query,
            answer=f"{sample.expected_answer} [1]",
            grounded=True,
            confidence_score=0.94,
            citations=[1],
            referenced_documents=[doc],
            stage_latencies=PipelineStageLatency(
                retrieval_ms=30.0,
                reranking_ms=20.0,
                generation_ms=90.0,
                total_ms=140.0,
            ),
            request_id=request_id or "eval-baseline",
            model="mock-gpt-4o",
        )

    mock_baseline_orch = MagicMock()
    mock_baseline_orch.execute = AsyncMock(side_effect=mock_execute_baseline)
    baseline_target = RAGOrchestratorTarget(mock_baseline_orch)

    # 4. Batch Execution
    runner = EvaluationRunner(target=baseline_target, metrics=metrics, max_concurrency=2)
    baseline_run = await runner.run_batch(dataset, run_name="baseline_v1_evaluation")

    assert baseline_run.sample_count == 3
    assert baseline_run.success_count == 3
    assert baseline_run.failure_count == 0
    assert baseline_run.failure_rate == 0.0
    assert len(baseline_run.aggregate_metrics) == 10
    assert baseline_run.aggregate_metrics["recall_at_3"].mean == 1.0
    assert baseline_run.aggregate_metrics["precision_at_3"].mean == round(1.0 / 3.0, 4)
    assert baseline_run.aggregate_metrics["mrr_at_5"].mean == 1.0
    assert baseline_run.aggregate_metrics["citation_accuracy"].mean == 1.0
    assert baseline_run.aggregate_metrics["answer_similarity_f1"].mean == 1.0
    assert baseline_run.latency_summary.mean_ms == 140.0

    # 5. Persist run to storage and verify retrieval
    store = FileEvaluationStore(base_directory=tmp_path / "eval_store")
    saved_id = await store.save_run(baseline_run)
    assert saved_id == baseline_run.run_id

    loaded_run = await store.get_run(saved_id)
    assert loaded_run is not None
    assert loaded_run.run_id == baseline_run.run_id
    assert loaded_run.aggregate_metrics["recall_at_3"].mean == 1.0

    # 6. Simulate candidate run with faster latency and compare
    async def mock_execute_candidate(
        query: str, request_id: str | None = None, **kwargs
    ) -> RAGResponse:
        resp = await mock_execute_baseline(query, request_id, **kwargs)
        # Faster latency: total 95ms
        return resp.model_copy(
            update={
                "stage_latencies": PipelineStageLatency(
                    retrieval_ms=20.0,
                    reranking_ms=10.0,
                    generation_ms=65.0,
                    total_ms=95.0,
                )
            }
        )

    mock_candidate_orch = MagicMock()
    mock_candidate_orch.execute = AsyncMock(side_effect=mock_execute_candidate)
    candidate_target = RAGOrchestratorTarget(mock_candidate_orch)

    candidate_runner = EvaluationRunner(target=candidate_target, metrics=metrics, max_concurrency=2)
    candidate_run = await candidate_runner.run_batch(dataset, run_name="candidate_v2_optimized")

    comparator = RunComparator()
    comparison = comparator.compare(baseline=baseline_run, candidate=candidate_run)

    assert comparison.baseline_run_id == baseline_run.run_id
    assert comparison.candidate_run_id == candidate_run.run_id
    assert comparison.mean_latency_delta_ms == -45.0  # 95 - 140 = -45ms
    assert comparison.failure_rate_delta == 0.0

    # 7. Generate Markdown reports
    run_report = ReportGenerator.generate_run_report(baseline_run)
    assert "# RAG Evaluation Report: baseline_v1_evaluation" in run_report
    assert "| **recall_at_3** | Automated | 1.000 |" in run_report
    assert "| **llm_answer_groundedness** | LLM Judge | 0.900 |" in run_report
    assert "| **Mean Latency** | 140.00 ms |" in run_report

    comparison_report = ReportGenerator.generate_comparison_report(comparison)
    assert "# RAG Evaluation Comparison Report" in comparison_report
    assert "Mean Latency Delta**: -45.00 ms" in comparison_report
    assert "⚪ UNCHANGED" in comparison_report
