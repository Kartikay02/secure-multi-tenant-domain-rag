"""Upload security guard protecting against zip bombs, double extensions, and traversal."""

import io
import zipfile
from pathlib import Path

from app.core.exceptions import FileValidationError, UnsupportedFileTypeError
from app.core.logging import get_logger

logger = get_logger("app.security.upload_guard")

DANGEROUS_EXTENSIONS: set[str] = {
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bat",
    ".cmd",
    ".sh",
    ".bash",
    ".ps1",
    ".vbs",
    ".vbe",
    ".js",
    ".jse",
    ".wsf",
    ".wsh",
    ".msc",
    ".py",
    ".pyc",
    ".pyd",
    ".php",
    ".phtml",
    ".jar",
    ".bin",
    ".elf",
    ".msi",
    ".scr",
    ".hta",
    ".cpl",
    ".inf",
    ".reg",
}

WINDOWS_RESERVED_NAMES: set[str] = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


class UploadSecurityGuard:
    """Validates files against malicious payload structures and decompression attacks."""

    def __init__(
        self,
        max_decompression_ratio: float = 20.0,
        max_uncompressed_bytes: int = 50 * 1024 * 1024,
        max_zip_entries: int = 1000,
    ) -> None:
        self.max_decompression_ratio = max_decompression_ratio
        self.max_uncompressed_bytes = max_uncompressed_bytes
        self.max_zip_entries = max_zip_entries

    def validate_filename_safety(self, filename: str) -> None:
        """Verify filename does not contain path traversal, null bytes, or dangerous double extensions."""
        if not filename or not filename.strip():
            raise FileValidationError("Uploaded filename cannot be empty.")

        if "\x00" in filename:
            raise FileValidationError("Filename contains forbidden null byte.")

        # SEC-04: Reject path traversal sequences immediately
        if ".." in filename or "/" in filename or "\\" in filename:
            logger.warning(f"Path traversal sequence detected in filename: '{filename}'")
            raise FileValidationError(
                f"Path traversal sequence detected in filename '{filename}'. Path traversal sequences ('..', '/', '\\') are forbidden."
            )

        clean_name = Path(filename).name
        stem = Path(clean_name).stem.lower()

        # Reject Windows reserved device names (e.g. CON.txt, AUX.pdf)
        if stem in WINDOWS_RESERVED_NAMES:
            raise FileValidationError(
                f"Filename '{clean_name}' uses a reserved system device name and cannot be processed."
            )

        # Inspect all extensions in case of double-extension spoofing (e.g. evil.exe.pdf)
        suffixes = [s.lower() for s in Path(clean_name).suffixes]
        # If the final extension itself is a known dangerous executable/script, reject as unsupported (HTTP 415)
        if suffixes and suffixes[-1] in DANGEROUS_EXTENSIONS:
            raise UnsupportedFileTypeError(
                extension=suffixes[-1],
                mime_type="unknown",
                allowed=[".pdf", ".docx", ".txt", ".md"],
            )

        # If any preceding extension is dangerous (e.g. evil.exe.pdf), reject as malicious double extension (HTTP 400)
        for s in suffixes[:-1]:
            if s in DANGEROUS_EXTENSIONS:
                raise FileValidationError(
                    f"File contains forbidden executable/script extension '{s}': '{clean_name}'"
                )

        # Disallow hidden files starting with a dot (e.g. .env, .htaccess)
        if clean_name.startswith(".") and len(suffixes) == 1 and suffixes[0] == clean_name:
            raise FileValidationError(f"Hidden system files are not permitted: '{clean_name}'")

    def check_zip_bomb(self, file_bytes: bytes) -> None:
        """Inspect archive metadata to detect zip bombs, decompression bombs, and entry traversal."""
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                infolist = zf.infolist()
                entry_count = len(infolist)

                if entry_count > self.max_zip_entries:
                    raise FileValidationError(
                        f"Archive rejected: contains {entry_count} entries, exceeding limit of {self.max_zip_entries}."
                    )

                total_uncompressed = 0
                total_compressed = 0

                for info in infolist:
                    # Traversal check inside entry names
                    entry_name = info.filename
                    if (
                        ".." in entry_name
                        or entry_name.startswith("/")
                        or entry_name.startswith("\\")
                        or ":" in entry_name
                    ):
                        raise FileValidationError(
                            f"Archive rejected: entry '{entry_name}' contains illegal path traversal sequences."
                        )

                    total_uncompressed += info.file_size
                    total_compressed += info.compress_size

                    # Individual entry uncompressed size check
                    if total_uncompressed > self.max_uncompressed_bytes:
                        raise FileValidationError(
                            f"Decompression bomb detected: uncompressed size exceeds limit of "
                            f"{self.max_uncompressed_bytes // (1024 * 1024)} MB."
                        )

                # Compression ratio check across archive
                effective_compressed = max(1, total_compressed)
                ratio = total_uncompressed / effective_compressed

                if ratio > self.max_decompression_ratio and total_uncompressed > (1024 * 1024):
                    logger.warning(
                        f"Zip bomb rejected: ratio {ratio:.1f}:1 exceeds safe threshold "
                        f"{self.max_decompression_ratio}:1 (uncompressed={total_uncompressed} bytes)"
                    )
                    raise FileValidationError(
                        f"Decompression bomb detected: archive compression ratio ({ratio:.1f}:1) "
                        f"exceeds safe threshold of {self.max_decompression_ratio}:1."
                    )

        except zipfile.BadZipFile as exc:
            raise FileValidationError(f"Corrupted or invalid zip archive: {exc}") from exc
