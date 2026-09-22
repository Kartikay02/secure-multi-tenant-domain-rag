"""File validation, safety sanitization, and cryptographic hashing."""

import hashlib
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from app.core.exceptions import (
    FileSizeLimitExceededError,
    FileValidationError,
    UnsupportedFileTypeError,
)
from app.core.logging import get_logger
from app.security.upload_guard import UploadSecurityGuard

logger = get_logger("app.ingestion.validator")

ALLOWED_EXTENSIONS: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".json": "application/json",
}


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a successful file validation."""

    filename: str
    extension: str
    detected_mime_type: str
    size_bytes: int
    content_hash: str


class FileValidator:
    """Validates file extensions, size limits, binary magic bytes, and sanitizes filenames."""

    def __init__(
        self,
        max_file_size_mb: int = 25,
        max_decompression_ratio: float = 20.0,
        max_uncompressed_size_mb: int = 50,
        max_zip_entries: int = 1000,
    ) -> None:
        self.max_bytes = max_file_size_mb * 1024 * 1024
        self.upload_guard = UploadSecurityGuard(
            max_decompression_ratio=max_decompression_ratio,
            max_uncompressed_bytes=max_uncompressed_size_mb * 1024 * 1024,
            max_zip_entries=max_zip_entries,
        )

    def sanitize_filename(self, raw_filename: str) -> str:
        """Sanitize filename against path traversal, control characters, and length limits."""
        # Deep security validation (reserved device names, dangerous double extensions, etc.)
        self.upload_guard.validate_filename_safety(raw_filename)

        # Take strictly the file basename
        name = Path(raw_filename).name

        # Strip dangerous sequences
        name = re.sub(r"\.\.+[/\\ ]", "", name)
        name = re.sub(r"[^\w\s\.-]", "_", name).strip()

        if not name:
            raise FileValidationError("Filename contains only invalid characters.")

        # Enforce maximum length
        if len(name) > 255:
            ext = Path(name).suffix
            stem = Path(name).stem[: 255 - len(ext)]
            name = f"{stem}{ext}"

        return name

    def _verify_magic_bytes(self, file_bytes: bytes, extension: str) -> str:
        """Verify file signatures (magic bytes) to prevent extension spoofing."""
        if extension == ".pdf":
            if not file_bytes.startswith(b"%PDF-"):
                raise FileValidationError("Invalid PDF file: Missing %PDF- file header signature.")
            return "application/pdf"

        if extension == ".docx":
            if not file_bytes.startswith(b"PK\x03\x04"):
                raise FileValidationError("Invalid DOCX file: Missing PK ZIP header signature.")

            # Decompression bomb and zip entry inspection
            self.upload_guard.check_zip_bomb(file_bytes)

            try:
                with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                    if "[Content_Types].xml" not in zf.namelist():
                        raise FileValidationError(
                            "Invalid DOCX file: Missing [Content_Types].xml manifest."
                        )
            except zipfile.BadZipFile as exc:
                raise FileValidationError(f"Corrupted DOCX archive: {exc}") from exc
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

        if extension in (".txt", ".md", ".csv", ".json"):
            # Plain text, markdown, CSV, and JSON must not contain null bytes
            if b"\x00" in file_bytes[:4096]:
                raise FileValidationError(f"Binary content detected in {extension} file.")
            try:
                file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    file_bytes.decode("latin-1")
                except UnicodeDecodeError as exc:
                    raise FileValidationError(
                        f"Cannot decode text in {extension} file: {exc}"
                    ) from exc
            if extension == ".md":
                return "text/markdown"
            if extension == ".csv":
                return "text/csv"
            if extension == ".json":
                return "application/json"
            return "text/plain"

        raise UnsupportedFileTypeError(
            extension=extension,
            mime_type="unknown",
            allowed=list(ALLOWED_EXTENSIONS.keys()),
        )

    def validate(self, file_bytes: bytes, raw_filename: str) -> ValidationResult:
        """Execute full validation suite on uploaded file bytes."""
        size_bytes = len(file_bytes)
        if size_bytes == 0:
            raise FileValidationError("Uploaded file is empty (0 bytes).")

        if size_bytes > self.max_bytes:
            raise FileSizeLimitExceededError(size_bytes, self.max_bytes)

        safe_filename = self.sanitize_filename(raw_filename)
        extension = Path(safe_filename).suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                extension=extension or "none",
                mime_type="unknown",
                allowed=list(ALLOWED_EXTENSIONS.keys()),
            )

        detected_mime = self._verify_magic_bytes(file_bytes, extension)
        content_hash = hashlib.sha256(file_bytes).hexdigest()

        return ValidationResult(
            filename=safe_filename,
            extension=extension,
            detected_mime_type=detected_mime,
            size_bytes=size_bytes,
            content_hash=content_hash,
        )
