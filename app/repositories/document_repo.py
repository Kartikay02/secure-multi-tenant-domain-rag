"""Document repository implementation."""

import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.exceptions import DatabaseError, DuplicateEntityError
from app.core.logging import get_logger
from app.models.document import Document, DocumentMetadata
from app.models.version import DocumentVersion
from app.repositories.base import BaseSQLAlchemyRepository
from app.repositories.interfaces.document import DocumentRepositoryProtocol

logger = get_logger("app.repository.document")


class SQLAlchemyDocumentRepository(BaseSQLAlchemyRepository[Document], DocumentRepositoryProtocol):
    """PostgreSQL data access implementation for Documents."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session=session, entity_class=Document)

    async def get_by_source(self, source: str) -> Document | None:
        """Lookup a document by its source URI."""
        try:
            query = select(Document).where(Document.source == source)
            result = await self.session.execute(query)
            return result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            logger.error(f"Error fetching document by source '{source}': {exc}")
            raise DatabaseError(f"Database error looking up document by source: {exc}") from exc

    async def get_with_versions(self, document_id: uuid.UUID) -> Document | None:
        """Fetch a document eagerly loading all versions and metadata (prevents N+1 queries)."""
        try:
            query = (
                select(Document)
                .where(Document.id == document_id)
                .options(
                    selectinload(Document.versions),
                    selectinload(Document.metadata_entries),
                )
            )
            result = await self.session.execute(query)
            return result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            logger.error(f"Error fetching document with versions {document_id}: {exc}")
            raise DatabaseError(f"Database error fetching document versions: {exc}") from exc

    async def create_with_version(
        self, document: Document, initial_version: DocumentVersion
    ) -> Document:
        """Atomically persist a document and its initial version in one transaction."""
        try:
            self.session.add(document)
            await self.session.flush()

            initial_version.document_id = document.id
            self.session.add(initial_version)
            await self.session.flush()

            return document
        except IntegrityError as exc:
            await self.session.rollback()
            logger.warning(f"Integrity violation creating document and version: {exc}")
            raise DuplicateEntityError(
                entity_name="Document/DocumentVersion",
                identifier=document.source,
            ) from exc
        except SQLAlchemyError as exc:
            await self.session.rollback()
            logger.error(f"Error in atomic document+version creation: {exc}")
            raise DatabaseError(f"Failed to create document with initial version: {exc}") from exc

    async def add_metadata(
        self, document_id: uuid.UUID, key: str, value: str, value_type: str = "string"
    ) -> DocumentMetadata:
        """Add or update a typed metadata key-value pair for a document."""
        try:
            metadata = DocumentMetadata(
                document_id=document_id,
                key=key,
                value=value,
                value_type=value_type,
            )
            self.session.add(metadata)
            await self.session.flush()
            return metadata
        except IntegrityError as exc:
            await self.session.rollback()
            logger.warning(f"Duplicate metadata key '{key}' for document {document_id}: {exc}")
            raise DuplicateEntityError(
                entity_name="DocumentMetadata",
                identifier=f"{document_id}:{key}",
            ) from exc
        except SQLAlchemyError as exc:
            await self.session.rollback()
            logger.error(f"Error adding metadata to document {document_id}: {exc}")
            raise DatabaseError(f"Failed to add document metadata: {exc}") from exc

    async def list_by_tenant(
        self, tenant_id: str, offset: int = 0, limit: int = 50
    ) -> list[Document]:
        """Fetch a paginated list of documents scoped to a specific tenant."""
        try:
            query = select(Document).order_by(Document.created_at.desc())
            result = await self.session.execute(query)
            all_docs = result.scalars().all()
            filtered = [
                d for d in all_docs if d.metadata_json.get("tenant_id", "default") == tenant_id
            ]
            return filtered[offset : offset + limit]
        except SQLAlchemyError as exc:
            logger.error(f"Failed to list documents for tenant '{tenant_id}': {exc}")
            raise DatabaseError(f"Error listing documents for tenant: {exc}") from exc

    async def count_by_tenant(self, tenant_id: str) -> int:
        """Count total documents belonging to a specific tenant."""
        try:
            query = select(Document)
            result = await self.session.execute(query)
            all_docs = result.scalars().all()
            return sum(
                1 for d in all_docs if d.metadata_json.get("tenant_id", "default") == tenant_id
            )
        except SQLAlchemyError as exc:
            logger.error(f"Failed to count documents for tenant '{tenant_id}': {exc}")
            raise DatabaseError(f"Error counting documents for tenant: {exc}") from exc
