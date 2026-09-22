"""Local filesystem implementation of FileStorageProtocol."""

import asyncio
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from app.core.exceptions import StorageError
from app.core.logging import get_logger
from app.rag.storage.interfaces import FileStorageProtocol

logger = get_logger("app.storage.local")


class LocalStorageService(FileStorageProtocol):
    """Stores files on the local filesystem with date-partitioned paths and atomic writes."""

    def __init__(self, base_dir: Path | str = "data/uploads") -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized LocalStorageService at: {self.base_dir}")

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent directory traversal and illegal characters."""
        if "\x00" in filename:
            raise StorageError("Invalid filename containing forbidden null byte.")
        clean = Path(filename).name
        clean = re.sub(r"[^\w\.-]", "_", clean)
        # Avoid reserved Windows device names
        stem = Path(clean).stem.lower()
        if stem in {"con", "prn", "aux", "nul", "com1", "com2", "com3", "com4", "lpt1", "lpt2"}:
            clean = f"safe_{clean}"
        return clean or "unnamed_document"

    def _resolve_uri_to_path(self, storage_uri: str) -> Path:
        """Convert a file:// URI or relative path to a validated local Path."""
        if "\x00" in storage_uri:
            raise StorageError("Invalid storage URI containing forbidden null byte.")

        if storage_uri.startswith("file://"):
            parsed = urlparse(storage_uri)
            # url2pathname handles drive letters and percent encoding (%20) on Windows and POSIX
            path_str = url2pathname(parsed.path)
        else:
            path_str = storage_uri

        target_path = Path(path_str).resolve()
        # Security check: ensure path does not escape base directory
        try:
            target_path.relative_to(self.base_dir)
        except ValueError as exc:
            logger.error(f"Path traversal attempt detected: {storage_uri}")
            raise StorageError(
                f"Access denied: URI '{storage_uri}' is outside storage root."
            ) from exc

        return target_path

    async def save(self, file_bytes: bytes, filename: str, content_type: str | None = None) -> str:
        """Atomically persist file bytes to date-partitioned directory."""
        safe_name = self._sanitize_filename(filename)
        date_folder = datetime.now(UTC).strftime("%Y/%m/%d")
        partition_dir = self.base_dir / date_folder
        unique_name = f"{uuid.uuid4().hex}_{safe_name}"
        destination = partition_dir / unique_name
        tmp_destination = partition_dir / f".{unique_name}.tmp"

        def _sync_write() -> Path:
            partition_dir.mkdir(parents=True, exist_ok=True)
            with open(tmp_destination, "wb") as f:
                f.write(file_bytes)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_destination, destination)
            return destination

        try:
            saved_path = await asyncio.to_thread(_sync_write)
            uri = saved_path.as_uri()
            logger.info(f"Successfully stored file ({len(file_bytes)} bytes) at: {uri}")
            return uri
        except Exception as exc:
            if tmp_destination.exists():
                tmp_destination.unlink(missing_ok=True)
            logger.error(f"Failed to persist file '{filename}' to local disk: {exc}")
            raise StorageError(f"Failed to store file '{filename}': {exc}") from exc

    async def read(self, storage_uri: str) -> bytes:
        """Read bytes from storage URI asynchronously."""
        target_path = self._resolve_uri_to_path(storage_uri)

        def _sync_read() -> bytes:
            if not target_path.is_file():
                raise StorageError(f"File not found at URI: {storage_uri}")
            with open(target_path, "rb") as f:
                return f.read()

        try:
            return await asyncio.to_thread(_sync_read)
        except StorageError:
            raise
        except Exception as exc:
            logger.error(f"Error reading file at URI '{storage_uri}': {exc}")
            raise StorageError(f"Could not read file from storage: {exc}") from exc

    async def delete(self, storage_uri: str) -> bool:
        """Delete file at storage URI asynchronously."""
        try:
            target_path = self._resolve_uri_to_path(storage_uri)

            def _sync_delete() -> bool:
                if target_path.is_file():
                    target_path.unlink()
                    return True
                return False

            return await asyncio.to_thread(_sync_delete)
        except Exception as exc:
            logger.error(f"Error deleting file at URI '{storage_uri}': {exc}")
            raise StorageError(f"Could not delete file from storage: {exc}") from exc

    async def exists(self, storage_uri: str) -> bool:
        """Check if file exists at storage URI."""
        try:
            target_path = self._resolve_uri_to_path(storage_uri)
            return await asyncio.to_thread(target_path.is_file)
        except Exception:
            return False
