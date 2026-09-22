"""Document repository protocol interface."""

import uuid
from typing import Protocol

from app.models.document import Document, DocumentMetadata
from app.models.version import DocumentVersion
from app.repositories.interfaces.base import BaseRepositoryProtocol


class DocumentRepositoryProtocol(BaseRepositoryProtocol[Document], Protocol):
    """Data access operations specific to Document entities."""

    async def get_by_source(self, source: str) -> Document | None:
        """Lookup a document by its source URI or origin path."""
        ...

    async def get_with_versions(self, document_id: uuid.UUID) -> Document | None:
        """Fetch a document eagerly loading all versions and metadata."""
        ...

    async def create_with_version(
        self, document: Document, initial_version: DocumentVersion
    ) -> Document:
        """Atomically persist a document and its initial version in one transaction."""
        ...

    async def add_metadata(
        self, document_id: uuid.UUID, key: str, value: str, value_type: str = "string"
    ) -> DocumentMetadata:
        """Add or update a typed metadata key-value pair for a document."""
        ...

    async def list_by_tenant(
        self, tenant_id: str, offset: int = 0, limit: int = 50
    ) -> list[Document]:
        """Fetch a paginated list of documents scoped to a specific tenant."""
        ...

    async def count_by_tenant(self, tenant_id: str) -> int:
        """Count total documents belonging to a specific tenant."""
        ...
