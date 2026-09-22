"""Unit tests for LocalStorageService."""

from pathlib import Path

import pytest

from app.core.exceptions import StorageError
from app.rag.storage.local import LocalStorageService


@pytest.mark.asyncio
async def test_local_storage_lifecycle(tmp_path: Path) -> None:
    """Verify save, read, exists, and delete on local filesystem."""
    storage = LocalStorageService(base_dir=tmp_path)
    sample_data = b"Arbitrary raw document content for storage testing."

    # 1. Save
    uri = await storage.save(sample_data, "invoice.pdf")
    assert uri.startswith("file://")
    assert await storage.exists(uri) is True

    # 2. Read
    retrieved = await storage.read(uri)
    assert retrieved == sample_data

    # 3. Delete
    deleted = await storage.delete(uri)
    assert deleted is True
    assert await storage.exists(uri) is False

    # 4. Read after delete raises StorageError
    with pytest.raises(StorageError):
        await storage.read(uri)


@pytest.mark.asyncio
async def test_local_storage_path_traversal_prevention(tmp_path: Path) -> None:
    """Verify attempts to access files outside the storage base directory are blocked."""
    storage = LocalStorageService(base_dir=tmp_path)

    # Attempt to read outside base directory
    illegal_uri = "file:///etc/passwd"
    with pytest.raises(StorageError) as exc_info:
        await storage.read(illegal_uri)
    assert "Access denied" in str(exc_info.value)
