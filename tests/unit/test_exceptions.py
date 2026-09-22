"""Unit tests for domain exception classes."""

from app.core.exceptions import (
    AppException,
    ConfigurationError,
    DuplicateEntityError,
    EntityNotFoundError,
    IngestionError,
    RateLimitExceededError,
)


def test_base_app_exception() -> None:
    """Verify base exception sets attributes correctly."""
    exc = AppException("Base error", error_code="CUSTOM_CODE", status_code=400, details={"k": "v"})
    assert exc.message == "Base error"
    assert exc.error_code == "CUSTOM_CODE"
    assert exc.status_code == 400
    assert exc.details == {"k": "v"}
    assert str(exc) == "Base error"


def test_entity_not_found_error() -> None:
    """Verify EntityNotFoundError structure."""
    exc = EntityNotFoundError(entity_name="Document", entity_id="doc-123")
    assert exc.status_code == 404
    assert exc.error_code == "NOT_FOUND"
    assert "Document with ID 'doc-123' not found." in exc.message
    assert exc.details["entity_name"] == "Document"
    assert exc.details["entity_id"] == "doc-123"


def test_duplicate_entity_error() -> None:
    """Verify DuplicateEntityError structure."""
    exc = DuplicateEntityError(entity_name="Chunk", identifier="chunk-hash-abc")
    assert exc.status_code == 409
    assert exc.error_code == "DUPLICATE_ENTITY"
    assert exc.details["identifier"] == "chunk-hash-abc"


def test_configuration_error() -> None:
    """Verify ConfigurationError default status and code."""
    exc = ConfigurationError("Invalid DB URI")
    assert exc.status_code == 500
    assert exc.error_code == "CONFIGURATION_ERROR"


def test_rate_limit_error() -> None:
    """Verify RateLimitExceededError structure and retry after details."""
    exc = RateLimitExceededError(retry_after_seconds=30)
    assert exc.status_code == 429
    assert exc.error_code == "RATE_LIMIT_EXCEEDED"
    assert exc.details["retry_after_seconds"] == 30


def test_ingestion_error() -> None:
    """Verify IngestionError defaults."""
    exc = IngestionError("PDF contains no extractable text", details={"pages": 0})
    assert exc.status_code == 422
    assert exc.error_code == "INGESTION_ERROR"
    assert exc.details == {"pages": 0}
