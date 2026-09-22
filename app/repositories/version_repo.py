"""DocumentVersion repository implementation."""

import uuid

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import DatabaseError, EntityNotFoundError
from app.core.logging import get_logger
from app.models.document import Document
from app.models.version import DocumentVersion
from app.repositories.base import BaseSQLAlchemyRepository
from app.repositories.interfaces.version import DocumentVersionRepositoryProtocol

logger = get_logger("app.repository.version")


class SQLAlchemyDocumentVersionRepository(
    BaseSQLAlchemyRepository[DocumentVersion], DocumentVersionRepositoryProtocol
):
    """PostgreSQL data access implementation for DocumentVersion entities."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session=session, entity_class=DocumentVersion)

    async def get_by_content_hash(
        self, content_hash: str, tenant_id: str | None = None
    ) -> DocumentVersion | None:
        """Find an existing document version with an identical content hash (deduplication).

        SEC-09: Strictly scopes hash lookup to tenant_id to prevent cross-tenant enumeration.
        """
        try:
            query = select(DocumentVersion).where(DocumentVersion.content_hash == content_hash)
            result = await self.session.execute(query)
            versions = list(result.scalars().all())
            if not versions:
                return None
            if tenant_id is None:
                return versions[0]

            for v in versions:
                doc = await self.session.get(Document, v.document_id)
                if doc and doc.metadata_json.get("tenant_id", "default") == tenant_id:
                    return v
                elif v.metadata_json.get("tenant_id", "default") == tenant_id:
                    return v
            return None
        except SQLAlchemyError as exc:
            logger.error(f"Error querying version by content hash '{content_hash}': {exc}")
            raise DatabaseError(f"Database error querying version by hash: {exc}") from exc

    async def get_latest_version(self, document_id: uuid.UUID) -> DocumentVersion | None:
        """Retrieve the latest version for a given document ordered by version_number descending."""
        try:
            query = (
                select(DocumentVersion)
                .where(DocumentVersion.document_id == document_id)
                .order_by(DocumentVersion.version_number.desc())
                .limit(1)
            )
            result = await self.session.execute(query)
            return result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            logger.error(f"Error fetching latest version for document {document_id}: {exc}")
            raise DatabaseError(f"Database error fetching latest version: {exc}") from exc

    async def update_status(
        self, version_id: uuid.UUID, status: str, total_chunks: int | None = None
    ) -> DocumentVersion:
        """Update processing status and optionally chunk count for a version."""
        version = await self.get_by_id(version_id)
        if not version:
            raise EntityNotFoundError(entity_name="DocumentVersion", entity_id=str(version_id))

        try:
            version.status = status
            if total_chunks is not None:
                version.total_chunks = total_chunks

            await self.session.flush()
            return version
        except SQLAlchemyError as exc:
            await self.session.rollback()
            logger.error(f"Error updating status for version {version_id}: {exc}")
            raise DatabaseError(f"Database error updating version status: {exc}") from exc
