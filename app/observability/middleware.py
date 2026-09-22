"""Observability middleware providing correlation ID injection, distributed trace context, and request latency tracking."""

import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import get_logger, set_correlation_id
from app.observability.metrics import (
    METRIC_ERRORS_TOTAL,
    METRIC_REQUEST_LATENCY_MS,
    METRIC_REQUESTS_TOTAL,
    get_metrics_collector,
)
from app.observability.sanitizer import Sanitizer
from app.observability.tracer import get_tracer, set_trace_id

logger = get_logger("app.observability.middleware")

CORRELATION_ID_HEADER = "X-Correlation-ID"
TRACE_ID_HEADER = "X-Trace-ID"
RESPONSE_TIME_HEADER = "X-Response-Time-Ms"


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Production HTTP middleware enforcing correlation, distributed tracing, metrics, and secret redaction."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # 1. Correlation & Trace Context Extraction
        corr_id = request.headers.get(CORRELATION_ID_HEADER)
        correlation_id = corr_id.strip() if corr_id and corr_id.strip() else uuid.uuid4().hex
        set_correlation_id(correlation_id)
        request.state.correlation_id = correlation_id

        incoming_trace = request.headers.get(TRACE_ID_HEADER)
        trace_id = (
            incoming_trace.strip() if incoming_trace and incoming_trace.strip() else correlation_id
        )
        set_trace_id(trace_id)
        request.state.trace_id = trace_id

        # 2. Timing & Tracing
        metrics = get_metrics_collector()
        tracer = get_tracer()
        start_time = time.perf_counter()

        method = request.method
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"

        # Sanitize query parameters before logging to avoid secret leakage in URLs
        query_str = Sanitizer.sanitize_string(str(request.url.query))
        query_suffix = f"?{query_str}" if query_str else ""
        logger.info(f"Incoming request: {method} {path}{query_suffix} from {client_ip}")

        span = tracer.start_span(
            name=f"HTTP {method} {path}",
            attributes={
                "http.method": method,
                "http.path": path,
                "http.client_ip": client_ip,
                "correlation_id": correlation_id,
                "trace_id": trace_id,
            },
        )

        try:
            response = await call_next(request)
            duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

            span.set_attribute("http.status_code", response.status_code)
            span.set_status("OK" if response.status_code < 400 else "ERROR")

            # Record metrics
            status_tag = str(response.status_code)
            metrics.increment_counter(
                METRIC_REQUESTS_TOTAL,
                value=1.0,
                tags={"endpoint": path, "method": method, "status": status_tag},
            )
            metrics.record_histogram(
                METRIC_REQUEST_LATENCY_MS,
                value=duration_ms,
                tags={"endpoint": path, "method": method},
            )

            if response.status_code >= 400:
                metrics.increment_counter(
                    METRIC_ERRORS_TOTAL,
                    value=1.0,
                    tags={
                        "stage": "http",
                        "status_code": status_tag,
                        "error_type": "HTTPStatusError",
                    },
                )

            # Response headers
            response.headers[CORRELATION_ID_HEADER] = correlation_id
            response.headers[TRACE_ID_HEADER] = trace_id
            response.headers[RESPONSE_TIME_HEADER] = str(duration_ms)

            logger.info(
                f"Completed request: {method} {path} -> status={response.status_code} in {duration_ms}ms"
            )
            return response

        except BaseException as exc:
            duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            span.record_exception(exc)
            span.set_status("ERROR", str(exc))

            metrics.increment_counter(
                METRIC_ERRORS_TOTAL,
                value=1.0,
                tags={"stage": "http", "status_code": "500", "error_type": type(exc).__name__},
            )
            metrics.record_histogram(
                METRIC_REQUEST_LATENCY_MS,
                value=duration_ms,
                tags={"endpoint": path, "method": method},
            )

            logger.error(
                f"Failed request: {method} {path} after {duration_ms}ms with unhandled error: {Sanitizer.sanitize_string(str(exc))}",
                exc_info=True,
            )
            raise exc
        finally:
            span.end()
