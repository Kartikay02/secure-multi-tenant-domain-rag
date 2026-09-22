"""Protocols and interfaces for production observability, metrics, tracing, and telemetry."""

from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol, runtime_checkable

from app.rag.orchestration.domain import RAGResponse


@runtime_checkable
class MetricsCollectorProtocol(Protocol):
    """Protocol for recording application counters, gauges, and latency histograms."""

    def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Increment a monotonically increasing counter metric."""
        ...

    def record_gauge(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Record an instantaneous numerical gauge value."""
        ...

    def record_histogram(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Record a sampled value into a distribution histogram (e.g. latency, size)."""
        ...

    def generate_prometheus_text(self) -> str:
        """Serialize current metric snapshot into Prometheus OpenMetrics text format."""
        ...

    def reset(self) -> None:
        """Reset metric registers."""
        ...


@runtime_checkable
class SpanProtocol(Protocol):
    """Protocol for an active tracing span."""

    def set_attribute(self, key: str, value: Any) -> None:
        """Attach a contextual key-value attribute to this span."""
        ...

    def set_status(self, status: str, description: str | None = None) -> None:
        """Set span execution outcome status (e.g. 'OK', 'ERROR')."""
        ...

    def record_exception(self, exc: BaseException) -> None:
        """Capture an exception and record error details on the span."""
        ...

    def end(self) -> None:
        """Conclude span timing and finalize record."""
        ...


@runtime_checkable
class TracerProtocol(Protocol):
    """Protocol for managing trace contexts and creating hierarchical spans."""

    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> SpanProtocol:
        """Begin an active tracing span."""
        ...

    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> AbstractAsyncContextManager[SpanProtocol]:
        """Async context manager creating and automatically completing a timed span."""
        ...


@runtime_checkable
class TelemetryHookProtocol(Protocol):
    """Protocol for dispatching RAG pipeline execution events to observability sinks."""

    def record_query_telemetry(self, response: RAGResponse) -> None:
        """Record metrics and telemetry from a completed RAG query execution."""
        ...

    def record_ingestion_telemetry(
        self,
        file_size: int,
        duration_ms: float,
        chunk_count: int,
        parser_name: str,
        status: str = "success",
    ) -> None:
        """Record metrics from document parsing and chunk ingestion."""
        ...

    def record_error_telemetry(
        self,
        stage: str,
        error: BaseException | str,
        status_code: int = 500,
    ) -> None:
        """Record pipeline stage exceptions and failure rates."""
        ...
