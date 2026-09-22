"""Unit tests for safe RFC 7807 error responses, secret redaction, and SQL injection resistance."""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.middleware.error_handler import register_exception_handlers
from app.observability.sanitizer import Sanitizer
from app.rag.retrieval.lexical import _compute_fallback_lexical_score


@pytest.mark.asyncio
async def test_unhandled_exception_does_not_leak_internals() -> None:
    """Verify unhandled internal exceptions return standardized RFC 7807 responses without stack traces."""
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/crash")
    async def crash() -> None:
        # Simulate unexpected crash with sensitive database credentials in exception message
        raise RuntimeError(
            "Fatal connection failure: postgresql://admin:SuperSecretPassword123@db.prod:5432/core"
        )

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        response = await client.get("/crash")

    assert response.status_code == 500
    data = response.json()

    # Verify RFC 7807 fields
    assert data["title"] == "Internal Server Error"
    assert data["status"] == 500
    assert data["error_code"] == "INTERNAL_SERVER_ERROR"
    assert data["detail"] == "An unexpected internal error occurred. Please contact system support."

    # Verify sensitive connection strings and stack traces are NOT in the response payload
    raw_response = response.text
    assert "SuperSecretPassword123" not in raw_response
    assert "db.prod" not in raw_response
    assert "Traceback" not in raw_response
    assert "RuntimeError" not in raw_response


def test_sql_injection_strings_treated_as_literal_tokens() -> None:
    """Verify SQL injection strings are safely tokenized and treated as pure lexical text."""
    malicious_sql_query = "' OR 1=1; DROP TABLE documents; --"
    chunk_text = "Standard document regarding database backup procedures."

    # Lexical score computation should run without errors and treat SQL syntax as inert search terms
    score = _compute_fallback_lexical_score(malicious_sql_query, chunk_text)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_secret_scrubbing_redacts_credentials() -> None:
    """Verify Sanitizer redacts API keys, bearer tokens, and passwords from logs."""
    sanitizer = Sanitizer()

    sensitive_text = (
        "Calling external API with sk-proj-1234567890abcdef1234567890abcdef and "
        "Authorization: Bearer my_jwt_access_token_1234567890"
    )
    scrubbed = sanitizer.sanitize_secrets(sensitive_text)

    assert "sk-proj-1234567890abcdef1234567890abcdef" not in scrubbed
    assert "my_jwt_access_token_1234567890" not in scrubbed
    assert "[REDACTED_API_KEY]" in scrubbed
    assert "[REDACTED]" in scrubbed
