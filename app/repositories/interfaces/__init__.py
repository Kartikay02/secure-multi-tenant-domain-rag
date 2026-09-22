"""Repository protocol interfaces for decoupled data access."""

from app.repositories.interfaces.base import BaseRepositoryProtocol
from app.repositories.interfaces.chunk import DocumentChunkRepositoryProtocol
from app.repositories.interfaces.document import DocumentRepositoryProtocol
from app.repositories.interfaces.job import IngestionJobRepositoryProtocol
from app.repositories.interfaces.version import DocumentVersionRepositoryProtocol

__all__ = [
    "BaseRepositoryProtocol",
    "DocumentChunkRepositoryProtocol",
    "DocumentRepositoryProtocol",
    "DocumentVersionRepositoryProtocol",
    "IngestionJobRepositoryProtocol",
]
