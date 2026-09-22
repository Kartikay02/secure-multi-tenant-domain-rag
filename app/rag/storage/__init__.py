"""Storage subsystem for raw document files."""

from app.rag.storage.interfaces import FileStorageProtocol
from app.rag.storage.local import LocalStorageService

__all__ = [
    "FileStorageProtocol",
    "LocalStorageService",
]
