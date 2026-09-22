"""Unit tests for RAGTelemetryHook and multi-stage telemetry metrics."""

from app.observability.hooks import RAGTelemetryHook
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
    InMemoryMetricsCollector,
)
from app.observability.tracer import InMemoryTracer
from app.rag.orchestration.domain import PipelineStageLatency, RAGResponse


def _create_rag_response(
    grounded: bool = True,
    insufficient_context: bool = False,
    prompt_tokens: int = 150,
    completion_tokens: int = 45,
) -> RAGResponse:
    return RAGResponse(
        query="Explain HNSW indexing",
        raw_query="Explain HNSW indexing",
        answer="HNSW builds multi-layer graphs for approximate nearest neighbor search.",
        grounded=grounded,
        confidence_score=0.92 if grounded else 0.40,
        insufficient_context=insufficient_context,
        fallback_applied=not grounded or insufficient_context,
        unsupported_claims=[] if grounded else ["Unsupported claim about quantum computers"],
        retrieved_chunks_count=10,
        context_chunks_count=5,
        context_tokens=350,
        stage_latencies=PipelineStageLatency(
            preprocessing_ms=1.5,
            retrieval_ms=25.0,
            reranking_ms=15.0,
            context_assembly_ms=2.0,
            generation_ms=80.0,
            validation_ms=10.0,
            total_ms=133.5,
        ),
        request_id="req-hook-test-1",
        model="gpt-4o",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def test_telemetry_hook_query_success() -> None:
    collector = InMemoryMetricsCollector()
    tracer = InMemoryTracer()
    hook = RAGTelemetryHook(metrics=collector, tracer=tracer)

    resp = _create_rag_response(grounded=True, insufficient_context=False)
    hook.record_query_telemetry(resp)

    # 1. Requests total counter
    assert (
        collector.get_counter(METRIC_REQUESTS_TOTAL, {"status": "success", "model": "gpt-4o"})
        == 1.0
    )

    # 2. Stage latencies
    ret_hist = collector.get_histogram_summary(METRIC_RETRIEVAL_LATENCY_MS, {"strategy": "hybrid"})
    assert ret_hist["count"] == 1.0
    assert ret_hist["mean"] == 25.0

    rerank_hist = collector.get_histogram_summary(
        METRIC_RERANKING_LATENCY_MS, {"fallback": "false"}
    )
    assert rerank_hist["mean"] == 15.0

    gen_hist = collector.get_histogram_summary(METRIC_GENERATION_LATENCY_MS, {"model": "gpt-4o"})
    assert gen_hist["mean"] == 80.0

    tot_hist = collector.get_histogram_summary(METRIC_REQUEST_LATENCY_MS, {"endpoint": "rag_query"})
    assert tot_hist["mean"] == 133.5

    # 3. Retrieval volume
    topk_hist = collector.get_histogram_summary(METRIC_RETRIEVAL_TOP_K, {"strategy": "hybrid"})
    assert topk_hist["mean"] == 10.0

    # 4. Token metrics
    assert (
        collector.get_counter(METRIC_TOKENS_TOTAL, {"type": "prompt", "model": "gpt-4o"}) == 150.0
    )
    assert (
        collector.get_counter(METRIC_TOKENS_TOTAL, {"type": "completion", "model": "gpt-4o"})
        == 45.0
    )

    # 5. Zero grounding failures
    assert collector.get_counter(METRIC_GROUNDING_FAILURES_TOTAL) == 0.0
    assert collector.get_counter(METRIC_INSUFFICIENT_CONTEXT_TOTAL) == 0.0


def test_telemetry_hook_grounding_failure_and_refusal() -> None:
    collector = InMemoryMetricsCollector()
    hook = RAGTelemetryHook(metrics=collector)

    resp_ungrounded = _create_rag_response(grounded=False, insufficient_context=True)
    hook.record_query_telemetry(resp_ungrounded)

    # Verify grounding failure incremented
    assert (
        collector.get_counter(
            METRIC_GROUNDING_FAILURES_TOTAL, {"fallback_applied": "true", "unsupported_claims": "1"}
        )
        == 1.0
    )
    assert (
        collector.get_counter(METRIC_INSUFFICIENT_CONTEXT_TOTAL, {"fallback_applied": "true"})
        == 1.0
    )


def test_telemetry_hook_ingestion_and_errors() -> None:
    collector = InMemoryMetricsCollector()
    hook = RAGTelemetryHook(metrics=collector)

    # Ingestion
    hook.record_ingestion_telemetry(
        file_size=2048,
        duration_ms=45.2,
        chunk_count=6,
        parser_name="markdown",
        status="success",
    )

    ingest_hist = collector.get_histogram_summary(
        METRIC_INGESTION_LATENCY_MS, {"parser": "markdown", "status": "success"}
    )
    assert ingest_hist["count"] == 1.0
    assert ingest_hist["mean"] == 45.2

    bytes_hist = collector.get_histogram_summary("rag_ingestion_bytes", {"parser": "markdown"})
    assert bytes_hist["mean"] == 2048.0

    assert collector.get_counter("rag_chunks_created_total", {"parser": "markdown"}) == 6.0

    # Error telemetry
    hook.record_error_telemetry(
        stage="retrieval", error=TimeoutError("PG connection dropped"), status_code=504
    )
    assert (
        collector.get_counter(
            METRIC_ERRORS_TOTAL,
            {"stage": "retrieval", "error_type": "TimeoutError", "status_code": "504"},
        )
        == 1.0
    )
