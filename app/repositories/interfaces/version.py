"""DocumentVersion repository protocol interface."""

import uuid
from typing import Protocol

from app.models.version import DocumentVersion
from app.repositories.interfaces.base import BaseRepositoryProtocol


class DocumentVersionRepositoryProtocol(BaseRepositoryProtocol[DocumentVersion], Protocol):
    """Data access operations specific to DocumentVersion revisions."""

    async def get_by_content_hash(
        self, content_hash: str, tenant_id: str | None = None
    ) -> DocumentVersion | None:
        """Find an existing version with matching content hash, optionally scoped by tenant."""
        ...

    async def get_latest_version(self, document_id: uuid.UUID) -> DocumentVersion | None:
        """Retrieve the most recent version for a given document."""
        ...

    async def update_status(
        self, version_id: uuid.UUID, status: str, total_chunks: int | None = None
    ) -> DocumentVersion:
        """Update processing status and optionally chunk count for a version."""
        ...
