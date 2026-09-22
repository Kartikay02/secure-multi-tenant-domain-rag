"""Unit tests for FileValidator, security sanitization, and content hashing."""

import hashlib
import io
import zipfile

import pytest

from app.core.exceptions import (
    FileSizeLimitExceededError,
    FileValidationError,
    UnsupportedFileTypeError,
)
from app.rag.ingestion.validator import FileValidator


def test_validator_valid_text_file() -> None:
    """Verify validation of a compliant text file."""
    validator = FileValidator(max_file_size_mb=10)
    raw_content = b"Simple document text."
    result = validator.validate(raw_content, "notes.txt")

    assert result.filename == "notes.txt"
    assert result.extension == ".txt"
    assert result.detected_mime_type == "text/plain"
    assert result.size_bytes == len(raw_content)
    assert result.content_hash == hashlib.sha256(raw_content).hexdigest()


def test_validator_file_size_exceeded() -> None:
    """Verify FileSizeLimitExceededError is raised when file exceeds threshold."""
    validator = FileValidator(max_file_size_mb=1)  # 1 MB
    oversized = b"0" * (1024 * 1024 + 50)
    with pytest.raises(FileSizeLimitExceededError):
        validator.validate(oversized, "big.txt")


def test_validator_empty_file_rejected() -> None:
    """Verify 0-byte file is rejected."""
    validator = FileValidator()
    with pytest.raises(FileValidationError) as exc:
        validator.validate(b"", "empty.txt")
    assert "empty" in str(exc.value)


def test_validator_unsupported_extension_rejected() -> None:
    """Verify unsupported file types (e.g. .exe, .sh) are rejected."""
    validator = FileValidator()
    with pytest.raises(UnsupportedFileTypeError):
        validator.validate(b"echo hello", "script.sh")


def test_validator_filename_sanitization() -> None:
    """Verify path traversal patterns and dangerous characters are rejected."""
    validator = FileValidator()

    # SEC-04: Traversal patterns rejected with FileValidationError
    with pytest.raises(FileValidationError):
        validator.sanitize_filename("../../etc/passwd.txt")

    with pytest.raises(FileValidationError):
        validator.sanitize_filename("..\\windows\\win.ini")

    # Null bytes rejected
    with pytest.raises(FileValidationError):
        validator.sanitize_filename("bad\x00file.txt")

    # Safe characters preserved and sanitized
    cleaned = validator.sanitize_filename("valid_report_2026.txt")
    assert cleaned == "valid_report_2026.txt"


def test_validator_pdf_magic_bytes() -> None:
    """Verify PDF magic bytes check enforces %PDF- header."""
    validator = FileValidator()

    valid_pdf_bytes = b"%PDF-1.7\n%some content"
    res = validator.validate(valid_pdf_bytes, "doc.pdf")
    assert res.detected_mime_type == "application/pdf"

    spoofed_pdf_bytes = b"Not a PDF header"
    with pytest.raises(FileValidationError) as exc:
        validator.validate(spoofed_pdf_bytes, "fake.pdf")
    assert "Missing %PDF-" in str(exc.value)


def test_validator_docx_magic_bytes() -> None:
    """Verify DOCX validation checks both ZIP signature and [Content_Types].xml manifest."""
    validator = FileValidator()

    # Create in-memory valid zip with [Content_Types].xml
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types></Types>")
    valid_docx_bytes = buffer.getvalue()

    res = validator.validate(valid_docx_bytes, "document.docx")
    assert "wordprocessingml" in res.detected_mime_type

    # Spoofed docx missing manifest
    bad_zip_buffer = io.BytesIO()
    with zipfile.ZipFile(bad_zip_buffer, "w") as zf:
        zf.writestr("some_random_file.txt", "data")
    with pytest.raises(FileValidationError) as exc:
        validator.validate(bad_zip_buffer.getvalue(), "bad.docx")
    assert "Missing [Content_Types].xml" in str(exc.value)
