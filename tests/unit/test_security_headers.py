"""Unit tests for OWASP security response headers middleware."""

import pytest
from fastapi import FastAPI, Response
from httpx import ASGITransport, AsyncClient

from app.security.headers import DEFAULT_SECURITY_HEADERS, SecurityHeadersMiddleware


@pytest.mark.asyncio
async def test_security_headers_injected_on_responses() -> None:
    """Verify standard security headers are injected on all outgoing HTTP responses."""
    test_app = FastAPI()
    test_app.add_middleware(SecurityHeadersMiddleware)

    @test_app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"status": "ok"}

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        response = await client.get("/ping")

    assert response.status_code == 200
    for header_name, expected_value in DEFAULT_SECURITY_HEADERS.items():
        assert header_name in response.headers
        assert response.headers[header_name] == expected_value


@pytest.mark.asyncio
async def test_security_headers_can_be_disabled() -> None:
    """Verify security headers middleware honors enabled=False toggle."""
    test_app = FastAPI()
    test_app.add_middleware(SecurityHeadersMiddleware, enabled=False)

    @test_app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"status": "ok"}

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        response = await client.get("/ping")

    assert response.status_code == 200
    assert "X-Frame-Options" not in response.headers
    assert "Content-Security-Policy" not in response.headers


@pytest.mark.asyncio
async def test_security_headers_preserves_custom_headers() -> None:
    """Verify custom existing response headers are not overwritten."""
    test_app = FastAPI()
    test_app.add_middleware(SecurityHeadersMiddleware)

    @test_app.get("/custom")
    async def custom() -> Response:
        from starlette.responses import JSONResponse

        return JSONResponse(
            content={"status": "custom"},
            headers={"X-Frame-Options": "SAMEORIGIN"},
        )

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        response = await client.get("/custom")

    assert response.status_code == 200
    # Explicitly set header on endpoint is preserved
    assert response.headers["X-Frame-Options"] == "SAMEORIGIN"
    # Other security headers are still injected
    assert response.headers["X-Content-Type-Options"] == "nosniff"


@pytest.mark.asyncio
async def test_hsts_omitted_on_plain_http() -> None:
    """Verify HSTS (Strict-Transport-Security) is strictly omitted on plain HTTP."""
    test_app = FastAPI()
    test_app.add_middleware(SecurityHeadersMiddleware)

    @test_app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"status": "ok"}

    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://insecure-test"
    ) as client:
        response = await client.get("/ping")

    assert response.status_code == 200
    assert "Strict-Transport-Security" not in response.headers


@pytest.mark.asyncio
async def test_hsts_emitted_on_true_https() -> None:
    """Verify HSTS is emitted when request arrives over true HTTPS."""
    test_app = FastAPI()
    test_app.add_middleware(SecurityHeadersMiddleware)

    @test_app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"status": "ok"}

    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="https://secure-test"
    ) as client:
        response = await client.get("/ping")

    assert response.status_code == 200
    assert "Strict-Transport-Security" in response.headers
    assert response.headers["Strict-Transport-Security"] == "max-age=31536000; includeSubDomains"


@pytest.mark.asyncio
async def test_hsts_emitted_via_trusted_proxy() -> None:
    """Verify HSTS is emitted when request arrives from a trusted proxy with X-Forwarded-Proto: https."""
    test_app = FastAPI()
    # 127.0.0.1 is in default trusted proxies
    test_app.add_middleware(SecurityHeadersMiddleware, trusted_proxies=["127.0.0.1", "10.0.0.1"])

    @test_app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"status": "ok"}

    transport = ASGITransport(app=test_app, client=("10.0.0.1", 12345))
    async with AsyncClient(transport=transport, base_url="http://internal-test") as client:
        response = await client.get("/ping", headers={"X-Forwarded-Proto": "https"})

    assert response.status_code == 200
    assert "Strict-Transport-Security" in response.headers


@pytest.mark.asyncio
async def test_hsts_rejected_via_untrusted_proxy_spoofing() -> None:
    """Verify spoofed X-Forwarded-Proto: https from untrusted hosts does NOT trigger HSTS."""
    test_app = FastAPI()
    test_app.add_middleware(SecurityHeadersMiddleware, trusted_proxies=["10.0.0.1"])

    @test_app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"status": "ok"}

    # Client IP 198.51.100.23 is not in trusted_proxies
    transport = ASGITransport(app=test_app, client=("198.51.100.23", 54321))
    async with AsyncClient(transport=transport, base_url="http://internal-test") as client:
        response = await client.get("/ping", headers={"X-Forwarded-Proto": "https"})

    assert response.status_code == 200
    assert "Strict-Transport-Security" not in response.headers
