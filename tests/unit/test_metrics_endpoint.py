"""Unit tests for Prometheus metrics scraping endpoints."""

import pytest
from httpx import AsyncClient

from app.observability.metrics import (
    METRIC_REQUESTS_TOTAL,
    InMemoryMetricsCollector,
    get_metrics_collector,
)


@pytest.mark.asyncio
async def test_metrics_scrape_endpoint(async_client: AsyncClient) -> None:
    collector = get_metrics_collector()
    assert isinstance(collector, InMemoryMetricsCollector)
    collector.reset()
    collector.increment_counter(METRIC_REQUESTS_TOTAL, 10.0, {"endpoint": "/api/v1/query"})

    # 1. Test root /metrics endpoint
    resp_root = await async_client.get("/metrics")
    assert resp_root.status_code == 200
    assert "text/plain" in resp_root.headers["content-type"]
    text_root = resp_root.text
    assert "# HELP rag_requests_total" in text_root
    assert "# TYPE rag_requests_total counter" in text_root
    assert 'rag_requests_total{endpoint="/api/v1/query"}' in text_root

    # 2. Test /api/v1/metrics endpoint
    resp_v1 = await async_client.get("/api/v1/metrics")
    assert resp_v1.status_code == 200
    assert "text/plain" in resp_v1.headers["content-type"]
    assert "# HELP rag_requests_total" in resp_v1.text
