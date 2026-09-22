"""Unit tests for ObservabilityMiddleware HTTP request tracking and header propagation."""

import pytest
from httpx import AsyncClient

from app.observability.metrics import (
    METRIC_ERRORS_TOTAL,
    METRIC_REQUEST_LATENCY_MS,
    METRIC_REQUESTS_TOTAL,
    InMemoryMetricsCollector,
    get_metrics_collector,
)


@pytest.mark.asyncio
async def test_observability_middleware_headers_and_metrics(async_client: AsyncClient) -> None:
    collector = get_metrics_collector()
    assert isinstance(collector, InMemoryMetricsCollector)
    collector.reset()

    # Request with custom correlation ID
    response = await async_client.get(
        "/health",
        headers={"X-Correlation-ID": "custom-corr-999"},
    )
    assert response.status_code == 200

    # Verify response headers
    assert response.headers["X-Correlation-ID"] == "custom-corr-999"
    assert "X-Trace-ID" in response.headers
    assert "X-Response-Time-Ms" in response.headers
    assert float(response.headers["X-Response-Time-Ms"]) >= 0.0

    # Verify metrics recorded
    assert (
        collector.get_counter(
            METRIC_REQUESTS_TOTAL, {"endpoint": "/health", "method": "GET", "status": "200"}
        )
        == 1.0
    )
    hist = collector.get_histogram_summary(
        METRIC_REQUEST_LATENCY_MS, {"endpoint": "/health", "method": "GET"}
    )
    assert hist["count"] == 1.0


@pytest.mark.asyncio
async def test_observability_middleware_records_error_metrics(async_client: AsyncClient) -> None:
    collector = get_metrics_collector()
    assert isinstance(collector, InMemoryMetricsCollector)
    collector.reset()

    # 404 endpoint request
    response = await async_client.get("/non-existent-path-for-testing")
    assert response.status_code == 404

    # Verify error metrics incremented
    assert (
        collector.get_counter(
            METRIC_REQUESTS_TOTAL,
            {"endpoint": "/non-existent-path-for-testing", "method": "GET", "status": "404"},
        )
        == 1.0
    )
    assert (
        collector.get_counter(
            METRIC_ERRORS_TOTAL,
            {"stage": "http", "status_code": "404", "error_type": "HTTPStatusError"},
        )
        == 1.0
    )
