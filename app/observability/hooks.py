"""Telemetry hooks bridging RAG pipeline stages and ingestion workflows with metrics and tracing."""

from app.observability.interfaces import (
    MetricsCollectorProtocol,
    TelemetryHookProtocol,
    TracerProtocol,
)
from app.observability.metrics import (
    METRIC_ERRORS_TOTAL,
    METRIC_GENERATION_LATENCY_MS,
    METRIC_GROUNDING_FAILURES_TOTAL,
    METRIC_INGESTION_LATENCY_MS,
    METRIC_INSUFFICIENT_CONTEXT_TOTAL,
    METRIC_REQUEST_LATENCY_MS,
    METRIC_REQUESTS_TOTAL,
    METRIC_RERANKING_LATENCY_MS,
    METRIC_RETRIEVAL_LATENCY_MS,
    METRIC_RETRIEVAL_TOP_K,
    METRIC_TOKENS_TOTAL,
    get_metrics_collector,
)
from app.observability.tracer import get_tracer
from app.rag.orchestration.domain import RAGResponse


class RAGTelemetryHook(TelemetryHookProtocol):
    """Production telemetry event dispatcher updating counters, distributions, and spans."""

    def __init__(
        self,
        metrics: MetricsCollectorProtocol | None = None,
        tracer: TracerProtocol | None = None,
    ) -> None:
        self.metrics = metrics or get_metrics_collector()
        self.tracer = tracer or get_tracer()

    def record_query_telemetry(self, response: RAGResponse) -> None:
        """Record multi-stage telemetry metrics from completed RAG inference."""
        # 1. Total requests & overall latency
        self.metrics.increment_counter(
            METRIC_REQUESTS_TOTAL,
            value=1.0,
            tags={"status": "success", "model": response.model or "unknown"},
        )
        if response.stage_latencies.total_ms > 0:
            self.metrics.record_histogram(
                METRIC_REQUEST_LATENCY_MS,
                value=response.stage_latencies.total_ms,
                tags={"endpoint": "rag_query"},
            )

        # 2. Stage-level latencies
        if response.stage_latencies.retrieval_ms > 0:
            self.metrics.record_histogram(
                METRIC_RETRIEVAL_LATENCY_MS,
                value=response.stage_latencies.retrieval_ms,
                tags={"strategy": "hybrid"},
            )
        if response.stage_latencies.reranking_ms > 0:
            self.metrics.record_histogram(
                METRIC_RERANKING_LATENCY_MS,
                value=response.stage_latencies.reranking_ms,
                tags={"fallback": str(response.metadata.get("rerank_fallback", False)).lower()},
            )
        if response.stage_latencies.generation_ms > 0:
            self.metrics.record_histogram(
                METRIC_GENERATION_LATENCY_MS,
                value=response.stage_latencies.generation_ms,
                tags={"model": response.model or "unknown"},
            )

        # 3. Retrieval volume
        if response.retrieved_chunks_count > 0:
            self.metrics.record_histogram(
                METRIC_RETRIEVAL_TOP_K,
                value=float(response.retrieved_chunks_count),
                tags={"strategy": "hybrid"},
            )

        # 4. Token usage
        if response.prompt_tokens:
            self.metrics.increment_counter(
                METRIC_TOKENS_TOTAL,
                value=float(response.prompt_tokens),
                tags={"type": "prompt", "model": response.model or "unknown"},
            )
        if response.completion_tokens:
            self.metrics.increment_counter(
                METRIC_TOKENS_TOTAL,
                value=float(response.completion_tokens),
                tags={"type": "completion", "model": response.model or "unknown"},
            )

        # 5. Grounding failures & insufficient context flags
        if not response.grounded:
            self.metrics.increment_counter(
                METRIC_GROUNDING_FAILURES_TOTAL,
                value=1.0,
                tags={
                    "fallback_applied": str(response.fallback_applied).lower(),
                    "unsupported_claims": str(len(response.unsupported_claims)),
                },
            )

        if response.insufficient_context:
            self.metrics.increment_counter(
                METRIC_INSUFFICIENT_CONTEXT_TOTAL,
                value=1.0,
                tags={"fallback_applied": str(response.fallback_applied).lower()},
            )

    def record_ingestion_telemetry(
        self,
        file_size: int,
        duration_ms: float,
        chunk_count: int,
        parser_name: str,
        status: str = "success",
    ) -> None:
        """Record ingestion pipeline execution time and output statistics."""
        self.metrics.record_histogram(
            METRIC_INGESTION_LATENCY_MS,
            value=duration_ms,
            tags={"parser": parser_name, "status": status},
        )
        self.metrics.record_histogram(
            "rag_ingestion_bytes",
            value=float(file_size),
            tags={"parser": parser_name},
        )
        if chunk_count > 0:
            self.metrics.increment_counter(
                "rag_chunks_created_total",
                value=float(chunk_count),
                tags={"parser": parser_name},
            )

    def record_error_telemetry(
        self,
        stage: str,
        error: BaseException | str,
        status_code: int = 500,
    ) -> None:
        """Increment error counters tagged by failure stage and error class."""
        error_name = type(error).__name__ if isinstance(error, BaseException) else "PipelineError"
        self.metrics.increment_counter(
            METRIC_ERRORS_TOTAL,
            value=1.0,
            tags={
                "stage": stage,
                "error_type": error_name,
                "status_code": str(status_code),
            },
        )


_global_telemetry_hook: TelemetryHookProtocol = RAGTelemetryHook()


def get_telemetry_hook() -> TelemetryHookProtocol:
    """Retrieve global telemetry hook singleton."""
    return _global_telemetry_hook


def set_telemetry_hook(hook: TelemetryHookProtocol) -> None:
    """Override global telemetry hook instance."""
    global _global_telemetry_hook
    _global_telemetry_hook = hook
