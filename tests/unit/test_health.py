"""Unit tests for health endpoints, correlation ID middleware, and error formatting."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_liveness(async_client: AsyncClient) -> None:
    """Verify GET /api/v1/health returns 200 OK and valid health payload."""
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert data["environment"] == "test"
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_health_readiness(async_client: AsyncClient) -> None:
    """Verify GET /api/v1/health/ready returns 200 OK and dependency checks."""
    response = await async_client.get("/api/v1/health/ready")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ready"
    assert "checks" in data
    assert data["checks"]["api"] == "ok"


@pytest.mark.asyncio
async def test_correlation_id_generated_when_missing(async_client: AsyncClient) -> None:
    """Verify X-Correlation-ID header is generated and returned if not provided."""
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200
    assert "x-correlation-id" in response.headers
    assert len(response.headers["x-correlation-id"]) > 0


@pytest.mark.asyncio
async def test_correlation_id_preserved_when_provided(async_client: AsyncClient) -> None:
    """Verify incoming X-Correlation-ID header is preserved and echoed in response."""
    custom_corr_id = "test-client-trace-777"
    response = await async_client.get(
        "/api/v1/health",
        headers={"X-Correlation-ID": custom_corr_id},
    )
    assert response.status_code == 200
    assert response.headers.get("x-correlation-id") == custom_corr_id


@pytest.mark.asyncio
async def test_response_time_header_present(async_client: AsyncClient) -> None:
    """Verify X-Response-Time-Ms header is present in the response."""
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200
    assert "x-response-time-ms" in response.headers
    duration = float(response.headers["x-response-time-ms"])
    assert duration >= 0.0


@pytest.mark.asyncio
async def test_404_error_formatting(async_client: AsyncClient) -> None:
    """Verify 404 responses are formatted as RFC 7807 problem details."""
    response = await async_client.get("/api/v1/non-existent-endpoint")
    assert response.status_code == 404

    data = response.json()
    assert data["status"] == 404
    assert data["error_code"] == "HTTP_404"
    assert "correlation_id" in data
    assert "detail" in data
    assert "timestamp" in data
