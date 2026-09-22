"""Sanitizer and redaction utilities preventing secrets and sensitive document content leakage."""

import hashlib
import re
from typing import Any

# Sensitive key names to automatically redact in structured dictionaries
SENSITIVE_KEY_NAMES = {
    "password",
    "passwd",
    "secret",
    "token",
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "x-api-key",
    "private_key",
    "credentials",
    "bearer",
    "client_secret",
}

# Regex patterns matching embedded secrets in text strings
SECRET_PATTERNS = [
    # Bearer tokens and Authorization headers
    (
        re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[A-Za-z0-9_\-\.]{10,}", re.IGNORECASE),
        r"\1[REDACTED]",
    ),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9_\-\.]{15,}", re.IGNORECASE), r"\1[REDACTED]"),
    # Common API key formats (sk-..., key-..., etc.)
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"), "[REDACTED_API_KEY]"),
    (
        re.compile(
            r"(?i)(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?([A-Za-z0-9_\-\.]{6,})['\"]?"
        ),
        r"\g<0>",
    ),  # Handled via custom sub
    # Credit card numbers
    (re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b"), "[REDACTED_CARD_NUMBER]"),
    # Social Security numbers
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
]

ASSIGNMENT_SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|secret|token|password|passwd|client[_-]?secret)\s*([:=])\s*['\"]?([A-Za-z0-9_\-\.]{6,})['\"]?",
    re.IGNORECASE,
)


class Sanitizer:
    """Sanitizes text strings and structured log dictionaries."""

    @classmethod
    def sanitize_string(cls, text: str) -> str:
        """Scrub known secret patterns from string."""
        if not text:
            return ""

        result = text
        for pattern, replacement in SECRET_PATTERNS:
            result = pattern.sub(replacement, result)

        # Replace assignment matches like api_key="secret123"
        result = ASSIGNMENT_SECRET_PATTERN.sub(r"\1\2[REDACTED]", result)
        return result

    # Alias for convenience
    sanitize_secrets = sanitize_string

    @classmethod
    def redact_document_content(cls, content: str, max_preview: int = 100) -> str:
        """Safely redact bulk document content, replacing with metadata descriptor and bounded preview."""
        if not content:
            return ""
        if len(content) <= max_preview:
            return cls.sanitize_string(content)

        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        preview = cls.sanitize_string(content[:max_preview].strip())
        return f"[REDACTED_CONTENT: length={len(content)}, sha256={content_hash}, preview='{preview}...']"

    @classmethod
    def sanitize_dict(
        cls,
        data: dict[str, Any],
        max_string_len: int = 500,
    ) -> dict[str, Any]:
        """Recursively sanitize dictionary keys and values for log extras and tracing attributes."""
        sanitized: dict[str, Any] = {}

        for key, value in data.items():
            key_lower = str(key).lower()

            # If the key name itself indicates a secret
            if any(s in key_lower for s in SENSITIVE_KEY_NAMES):
                sanitized[key] = "[REDACTED]"
                continue

            # If the key contains document body or raw content
            if key_lower in {"content", "document_content", "raw_text", "file_bytes", "body"}:
                if isinstance(value, str):
                    sanitized[key] = cls.redact_document_content(value)
                elif isinstance(value, bytes):
                    sanitized[key] = f"[REDACTED_BYTES: length={len(value)}]"
                else:
                    sanitized[key] = "[REDACTED_PAYLOAD]"
                continue

            if isinstance(value, str):
                if len(value) > max_string_len:
                    # Bounded sanitization for oversized values
                    sanitized[key] = cls.sanitize_string(value[:max_string_len]) + "...[TRUNCATED]"
                else:
                    sanitized[key] = cls.sanitize_string(value)
            elif isinstance(value, dict):
                sanitized[key] = cls.sanitize_dict(value, max_string_len=max_string_len)
            elif isinstance(value, list):
                sanitized[key] = [
                    cls.sanitize_dict(item, max_string_len=max_string_len)
                    if isinstance(item, dict)
                    else (cls.sanitize_string(item) if isinstance(item, str) else item)
                    for item in value
                ]
            else:
                sanitized[key] = value

        return sanitized
