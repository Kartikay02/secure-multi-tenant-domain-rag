"""Storage abstraction protocols for raw file persistence."""

from typing import Protocol


class FileStorageProtocol(Protocol):
    """Abstract interface for raw document file storage (Local disk, S3, GCS, Azure Blob)."""

    async def save(self, file_bytes: bytes, filename: str, content_type: str | None = None) -> str:
        """Persist file bytes to the storage backend and return its canonical URI."""
        ...

    async def read(self, storage_uri: str) -> bytes:
        """Retrieve raw file bytes given its storage URI."""
        ...

    async def delete(self, storage_uri: str) -> bool:
        """Delete a file from storage given its URI. Return True if deleted, False if not found."""
        ...

    async def exists(self, storage_uri: str) -> bool:
        """Check whether a file exists at the specified storage URI."""
        ...
