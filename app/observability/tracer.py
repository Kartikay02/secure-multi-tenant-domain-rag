"""Distributed tracing abstraction and in-memory trace recorder."""

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.logging import get_correlation_id
from app.observability.interfaces import SpanProtocol, TracerProtocol
from app.observability.sanitizer import Sanitizer


class SpanData(BaseModel):
    """Immutable snapshot of a concluded tracing span."""

    model_config = ConfigDict(frozen=True)

    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    start_time: datetime
    end_time: datetime | None = None
    duration_ms: float = 0.0
    status: str = "OK"
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)


# Context variables for trace context propagation
_current_span_ctx: ContextVar["InMemorySpan | None"] = ContextVar("current_span", default=None)
_trace_id_ctx: ContextVar[str] = ContextVar("trace_id", default="")


def get_trace_id() -> str:
    """Retrieve active trace ID from context or correlation ID."""
    return _trace_id_ctx.get() or get_correlation_id() or ""


def set_trace_id(trace_id: str) -> None:
    """Set active trace ID in context."""
    _trace_id_ctx.set(trace_id)


class InMemorySpan(SpanProtocol):
    """Active span implementation capturing telemetry, attributes, and exceptions."""

    def __init__(
        self,
        name: str,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
        on_finish: Any = None,
    ) -> None:
        self.name = name
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.start_time = datetime.now(UTC)
        self._start_perf = time.perf_counter()
        self.end_time: datetime | None = None
        self.duration_ms: float = 0.0
        self.status = "OK"
        self.attributes: dict[str, Any] = Sanitizer.sanitize_dict(attributes or {})
        self.events: list[dict[str, Any]] = []
        self._on_finish = on_finish
        self._ended = False

    def set_attribute(self, key: str, value: Any) -> None:
        sanitized = Sanitizer.sanitize_dict({key: value})
        self.attributes.update(sanitized)

    def set_status(self, status: str, description: str | None = None) -> None:
        self.status = status
        if description:
            self.attributes["status_description"] = Sanitizer.sanitize_string(description)

    def record_exception(self, exc: BaseException) -> None:
        self.status = "ERROR"
        self.events.append(
            {
                "name": "exception",
                "timestamp": datetime.now(UTC).isoformat(),
                "exception_type": type(exc).__name__,
                "message": Sanitizer.sanitize_string(str(exc)),
            }
        )

    def end(self) -> None:
        if self._ended:
            return
        self._ended = True
        self.duration_ms = round((time.perf_counter() - self._start_perf) * 1000.0, 2)
        self.end_time = datetime.now(UTC)
        if self._on_finish:
            self._on_finish(self)

    def to_data(self) -> SpanData:
        return SpanData(
            name=self.name,
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            start_time=self.start_time,
            end_time=self.end_time,
            duration_ms=self.duration_ms,
            status=self.status,
            attributes=dict(self.attributes),
            events=list(self.events),
        )


class InMemoryTracer(TracerProtocol):
    """Context-propagating distributed tracer storing concluded spans in memory."""

    def __init__(self) -> None:
        self._spans: list[SpanData] = []

    @property
    def recorded_spans(self) -> list[SpanData]:
        return list(self._spans)

    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> SpanProtocol:
        parent_span = _current_span_ctx.get()
        if parent_span is not None:
            active_trace_id = parent_span.trace_id
        else:
            active_trace_id = get_trace_id() or uuid.uuid4().hex

        span = InMemorySpan(
            name=name,
            trace_id=active_trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=parent_span.span_id if parent_span else None,
            attributes=attributes,
            on_finish=self._record_span,
        )
        _current_span_ctx.set(span)
        return span

    def _record_span(self, span: InMemorySpan) -> None:
        self._spans.append(span.to_data())

    @asynccontextmanager
    async def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> AsyncIterator[SpanProtocol]:
        parent = _current_span_ctx.get()
        active_span = self.start_span(name, attributes)
        try:
            yield active_span
        except BaseException as exc:
            active_span.record_exception(exc)
            raise
        finally:
            active_span.end()
            _current_span_ctx.set(parent)

    def reset(self) -> None:
        """Clear recorded spans."""
        self._spans.clear()


class NoOpTracer(TracerProtocol):
    """No-op tracer for zero-overhead execution when tracing is disabled."""

    def start_span(self, name: str, attributes: dict[str, Any] | None = None) -> SpanProtocol:
        class _NoOpSpan(SpanProtocol):
            def set_attribute(self, key: str, value: Any) -> None:
                pass

            def set_status(self, status: str, description: str | None = None) -> None:
                pass

            def record_exception(self, exc: BaseException) -> None:
                pass

            def end(self) -> None:
                pass

        return _NoOpSpan()

    @asynccontextmanager
    async def span(
        self, name: str, attributes: dict[str, Any] | None = None
    ) -> AsyncIterator[SpanProtocol]:
        yield self.start_span(name, attributes)


_global_tracer: TracerProtocol = InMemoryTracer()


def get_tracer() -> TracerProtocol:
    """Retrieve global tracer instance."""
    return _global_tracer


def set_tracer(tracer: TracerProtocol) -> None:
    """Override global tracer instance."""
    global _global_tracer
    _global_tracer = tracer
