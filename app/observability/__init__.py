"""Production observability module exports."""

from app.observability.dashboard import (
    PRODUCTION_SLOS,
    PROMETHEUS_ALERT_RULES,
    get_grafana_dashboard_spec,
)
from app.observability.hooks import (
    RAGTelemetryHook,
    get_telemetry_hook,
    set_telemetry_hook,
)
from app.observability.interfaces import (
    MetricsCollectorProtocol,
    SpanProtocol,
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
    InMemoryMetricsCollector,
    NoOpMetricsCollector,
    get_metrics_collector,
    set_metrics_collector,
)
from app.observability.middleware import ObservabilityMiddleware
from app.observability.sanitizer import Sanitizer
from app.observability.tracer import (
    InMemorySpan,
    InMemoryTracer,
    NoOpTracer,
    SpanData,
    get_trace_id,
    get_tracer,
    set_trace_id,
    set_tracer,
)

__all__ = [
    # Interfaces
    "MetricsCollectorProtocol",
    "SpanProtocol",
    "TracerProtocol",
    "TelemetryHookProtocol",
    # Metrics
    "InMemoryMetricsCollector",
    "NoOpMetricsCollector",
    "get_metrics_collector",
    "set_metrics_collector",
    "METRIC_REQUESTS_TOTAL",
    "METRIC_ERRORS_TOTAL",
    "METRIC_REQUEST_LATENCY_MS",
    "METRIC_INGESTION_LATENCY_MS",
    "METRIC_RETRIEVAL_LATENCY_MS",
    "METRIC_RERANKING_LATENCY_MS",
    "METRIC_GENERATION_LATENCY_MS",
    "METRIC_TOKENS_TOTAL",
    "METRIC_RETRIEVAL_TOP_K",
    "METRIC_GROUNDING_FAILURES_TOTAL",
    "METRIC_INSUFFICIENT_CONTEXT_TOTAL",
    # Tracing
    "SpanData",
    "InMemorySpan",
    "InMemoryTracer",
    "NoOpTracer",
    "get_tracer",
    "set_tracer",
    "get_trace_id",
    "set_trace_id",
    # Sanitization
    "Sanitizer",
    # Hooks
    "RAGTelemetryHook",
    "get_telemetry_hook",
    "set_telemetry_hook",
    # Middleware
    "ObservabilityMiddleware",
    # Dashboards & Alerts
    "PRODUCTION_SLOS",
    "PROMETHEUS_ALERT_RULES",
    "get_grafana_dashboard_spec",
]
