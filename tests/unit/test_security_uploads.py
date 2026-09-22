"""Unit tests for malicious file upload defense, zip bombs, and filename sanitization."""

import io
import zipfile

import pytest

from app.core.exceptions import (
    FileSizeLimitExceededError,
    FileValidationError,
)
from app.rag.ingestion.validator import FileValidator
from app.security.upload_guard import UploadSecurityGuard


def _create_minimal_docx_bytes(extra_files: dict[str, bytes] | None = None) -> bytes:
    """Helper creating a syntactically valid minimal DOCX zip byte buffer."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", b'<?xml version="1.0"?><Types></Types>')
        zf.writestr("word/document.xml", b'<?xml version="1.0"?><w:document></w:document>')
        if extra_files:
            for fname, fcontent in extra_files.items():
                zf.writestr(fname, fcontent)
    return buf.getvalue()


def test_disguised_executable_as_pdf_rejected() -> None:
    """Verify an executable binary disguised with a .pdf extension is rejected."""
    validator = FileValidator()
    # Executable binary header (MZ) disguised as PDF
    fake_pdf = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00Not a real PDF"
    with pytest.raises(FileValidationError) as exc:
        validator.validate(fake_pdf, "invoice.pdf")
    assert "Missing %PDF-" in str(exc.value)


def test_dangerous_double_extension_rejected() -> None:
    """Verify files with disguised double extensions (e.g. report.exe.pdf) are blocked."""
    validator = FileValidator()
    dummy_pdf = b"%PDF-1.7\nSample document"

    dangerous_names = [
        "report.exe.pdf",
        "financials.bat.docx",
        "script.sh.txt",
        "payload.py.md",
        "invoice.dll.pdf",
        "update.vbs.txt",
    ]

    for name in dangerous_names:
        with pytest.raises(FileValidationError) as exc:
            validator.validate(dummy_pdf, name)
        assert "forbidden executable/script extension" in str(exc.value)


def test_windows_reserved_names_rejected() -> None:
    """Verify Windows system device names (CON, NUL, AUX, COM1) are rejected."""
    validator = FileValidator()
    dummy_pdf = b"%PDF-1.7\nValid pdf content"

    reserved = ["con.pdf", "NUL.docx", "aux.txt", "prn.md", "com1.pdf", "lpt1.txt"]
    for r in reserved:
        with pytest.raises(FileValidationError) as exc:
            validator.validate(dummy_pdf, r)
        assert "reserved system device name" in str(exc.value)


def test_null_byte_in_filename_rejected() -> None:
    """Verify null bytes in filenames are caught and rejected."""
    guard = UploadSecurityGuard()
    with pytest.raises(FileValidationError) as exc:
        guard.validate_filename_safety("legit.pdf\x00.exe")
    assert "null byte" in str(exc.value)


def test_docx_zip_entry_path_traversal_rejected() -> None:
    """Verify zip entries containing directory traversal paths are rejected."""
    guard = UploadSecurityGuard()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", b"<Types/>")
        zf.writestr("../../evil.sh", b"rm -rf /")

    with pytest.raises(FileValidationError) as exc:
        guard.check_zip_bomb(buf.getvalue())
    assert "illegal path traversal sequences" in str(exc.value)


def test_docx_decompression_bomb_ratio_rejected() -> None:
    """Verify docx archive with dangerous compression ratio (> 20:1) is rejected."""
    guard = UploadSecurityGuard(max_decompression_ratio=15.0)

    # 10MB of repetitive zeroes compresses into ~10KB (ratio > 500:1)
    large_payload = b"A" * (5 * 1024 * 1024)
    bomb_docx = _create_minimal_docx_bytes({"word/large_text.xml": large_payload})

    with pytest.raises(FileValidationError) as exc:
        guard.check_zip_bomb(bomb_docx)
    assert "Decompression bomb detected" in str(exc.value)


def test_docx_excessive_zip_entries_rejected() -> None:
    """Verify docx archive containing an excessive count of file entries is rejected."""
    guard = UploadSecurityGuard(max_zip_entries=50)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", b"<Types/>")
        for i in range(60):
            zf.writestr(f"entry_{i}.xml", b"<x/>")

    with pytest.raises(FileValidationError) as exc:
        guard.check_zip_bomb(buf.getvalue())
    assert "contains 61 entries, exceeding limit of 50" in str(exc.value)


def test_oversized_file_rejected() -> None:
    """Verify files exceeding max_file_size_mb are rejected."""
    validator = FileValidator(max_file_size_mb=1)
    two_mb_payload = b"A" * (2 * 1024 * 1024)
    with pytest.raises(FileSizeLimitExceededError):
        validator.validate(two_mb_payload, "oversized.txt")


def test_empty_file_rejected() -> None:
    """Verify 0-byte files are rejected."""
    validator = FileValidator()
    with pytest.raises(FileValidationError) as exc:
        validator.validate(b"", "empty.txt")
    assert "empty" in str(exc.value)


def test_valid_docx_passes_validation() -> None:
    """Verify legitimate docx file passes validation cleanly."""
    validator = FileValidator()
    valid_docx = _create_minimal_docx_bytes()
    res = validator.validate(valid_docx, "valid_document.docx")
    assert res.extension == ".docx"
    assert (
        res.detected_mime_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert res.size_bytes > 0
