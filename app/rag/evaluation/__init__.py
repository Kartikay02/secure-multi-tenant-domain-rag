"""RAG evaluation subsystem public API."""

from app.rag.evaluation.comparator import RunComparator
from app.rag.evaluation.domain import (
    AggregateMetricStats,
    EvaluationDataset,
    EvaluationPrediction,
    EvaluationRun,
    EvaluationRunSummary,
    EvaluationSample,
    EvaluationSampleResult,
    LatencySummary,
    MetricCategory,
    MetricDelta,
    MetricResult,
    RunComparisonResult,
    SampleRegression,
)
from app.rag.evaluation.interfaces import (
    EvaluationMetricProtocol,
    EvaluationStoreProtocol,
    EvaluationTargetProtocol,
    LLMJudgeProtocol,
)
from app.rag.evaluation.metrics import (
    AnswerGroundednessMetric,
    AnswerLexicalSimilarityMetric,
    CitationAccuracyMetric,
    ContextRelevanceMetric,
    LLMAnswerGroundednessJudge,
    LLMAnswerQualityJudge,
    LLMContextRelevanceJudge,
    MockLLMJudge,
    MRRMetric,
    PrecisionAtKMetric,
    RecallAtKMetric,
)
from app.rag.evaluation.reports import ReportGenerator
from app.rag.evaluation.runner import EvaluationRunner, RAGOrchestratorTarget
from app.rag.evaluation.store import FileEvaluationStore, InMemoryEvaluationStore

__all__ = [
    # Domain
    "EvaluationSample",
    "EvaluationDataset",
    "EvaluationPrediction",
    "MetricCategory",
    "MetricResult",
    "AggregateMetricStats",
    "LatencySummary",
    "EvaluationSampleResult",
    "EvaluationRun",
    "EvaluationRunSummary",
    "MetricDelta",
    "SampleRegression",
    "RunComparisonResult",
    # Interfaces
    "EvaluationMetricProtocol",
    "EvaluationTargetProtocol",
    "EvaluationStoreProtocol",
    "LLMJudgeProtocol",
    # Metrics
    "RecallAtKMetric",
    "PrecisionAtKMetric",
    "MRRMetric",
    "ContextRelevanceMetric",
    "AnswerGroundednessMetric",
    "CitationAccuracyMetric",
    "AnswerLexicalSimilarityMetric",
    "MockLLMJudge",
    "LLMContextRelevanceJudge",
    "LLMAnswerGroundednessJudge",
    "LLMAnswerQualityJudge",
    # Runner & Adapter
    "EvaluationRunner",
    "RAGOrchestratorTarget",
    # Store
    "FileEvaluationStore",
    "InMemoryEvaluationStore",
    # Comparator & Reports
    "RunComparator",
    "ReportGenerator",
]
