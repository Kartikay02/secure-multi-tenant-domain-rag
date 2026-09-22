"""Unit tests for secret scrubbing and document content sanitization."""

from app.observability.sanitizer import Sanitizer


def test_sanitize_string_secrets() -> None:
    # Bearer token
    msg = "Client authorized with Authorization: Bearer abc123xyz456secret"
    clean = Sanitizer.sanitize_string(msg)
    assert "abc123xyz456secret" not in clean
    assert "[REDACTED]" in clean

    # API key assignment
    msg2 = 'Connecting with api_key="secret_key_value_98765"'
    clean2 = Sanitizer.sanitize_string(msg2)
    assert "secret_key_value_98765" not in clean2
    assert "[REDACTED]" in clean2

    # sk- format API key
    msg3 = "OpenAI key sk-proj1234567890abcdef12345678 configured"
    clean3 = Sanitizer.sanitize_string(msg3)
    assert "sk-proj1234567890abcdef12345678" not in clean3
    assert "[REDACTED_API_KEY]" in clean3


def test_redact_document_content() -> None:
    short_content = "Short note"
    assert Sanitizer.redact_document_content(short_content) == "Short note"

    long_content = (
        "CONFIDENTIAL CUSTOMER RECORD\n"
        "Name: John Doe\n"
        "Account: 9988776655\n"
        "Diagnosis: Medical record text detailing internal medical symptoms..." * 5
    )

    redacted = Sanitizer.redact_document_content(long_content, max_preview=30)
    assert "CONFIDENTIAL" in redacted
    assert "length=" in redacted
    assert "sha256=" in redacted
    # Raw multi-sentence sensitive content is not dumped in full
    assert len(redacted) < len(long_content)


def test_sanitize_dict_nested() -> None:
    raw_payload = {
        "user_id": "usr-123",
        "password": "super_secret_password",
        "api_key": "sk-1234567890abcdef12345",
        "headers": {
            "Authorization": "Bearer token_abc_def_ghi_12345",
            "Content-Type": "application/json",
        },
        "content": "A massive 500-page operational manual text " * 10,
        "tags": ["prod", "auth=secret_token_12345"],
    }

    sanitized = Sanitizer.sanitize_dict(raw_payload)

    assert sanitized["user_id"] == "usr-123"
    assert sanitized["password"] == "[REDACTED]"
    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["headers"]["Authorization"] == "[REDACTED]"
    assert sanitized["headers"]["Content-Type"] == "application/json"
    assert "[REDACTED_CONTENT" in sanitized["content"]
