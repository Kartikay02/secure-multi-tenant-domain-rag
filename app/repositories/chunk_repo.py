"""DocumentChunk repository implementation."""

import uuid

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer

from app.core.exceptions import DatabaseError, DuplicateEntityError, EntityNotFoundError
from app.core.logging import get_logger
from app.models.chunk import DocumentChunk
from app.repositories.base import BaseSQLAlchemyRepository
from app.repositories.interfaces.chunk import DocumentChunkRepositoryProtocol

logger = get_logger("app.repository.chunk")


class SQLAlchemyDocumentChunkRepository(
    BaseSQLAlchemyRepository[DocumentChunk], DocumentChunkRepositoryProtocol
):
    """PostgreSQL data access implementation for DocumentChunk entities."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session=session, entity_class=DocumentChunk)

    async def bulk_create_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Atomically persist a batch of document chunks."""
        if not chunks:
            return []
        try:
            self.session.add_all(chunks)
            await self.session.flush()
            return chunks
        except IntegrityError as exc:
            await self.session.rollback()
            logger.warning(f"Integrity violation bulk-inserting chunks: {exc}")
            raise DuplicateEntityError(
                entity_name="DocumentChunkBatch",
                identifier=f"version_id={chunks[0].document_version_id}",
            ) from exc
        except SQLAlchemyError as exc:
            await self.session.rollback()
            logger.error(f"Database error bulk-inserting chunks: {exc}")
            raise DatabaseError(f"Database error bulk-inserting chunks: {exc}") from exc

    async def get_chunks_by_version(
        self, version_id: uuid.UUID, include_content: bool = True
    ) -> list[DocumentChunk]:
        """Retrieve all chunks for a document version.

        When include_content=False, defers loading large text payloads (saves memory and I/O).
        """
        try:
            query = (
                select(DocumentChunk)
                .where(DocumentChunk.document_version_id == version_id)
                .order_by(DocumentChunk.chunk_index.asc())
            )

            # Avoid loading large text content when only metadata/index inspection is required
            if not include_content:
                query = query.options(defer(DocumentChunk.content))

            result = await self.session.execute(query)
            return list(result.scalars().all())
        except SQLAlchemyError as exc:
            logger.error(f"Error fetching chunks for version {version_id}: {exc}")
            raise DatabaseError(f"Database error fetching chunks: {exc}") from exc

    async def count_by_version(self, version_id: uuid.UUID) -> int:
        """Count total chunks belonging to a document version."""
        try:
            query = select(func.count(DocumentChunk.id)).where(
                DocumentChunk.document_version_id == version_id
            )
            result = await self.session.execute(query)
            return int(result.scalar_one() or 0)
        except SQLAlchemyError as exc:
            logger.error(f"Error counting chunks for version {version_id}: {exc}")
            raise DatabaseError(f"Database error counting chunks: {exc}") from exc

    async def delete_chunks_by_version(self, version_id: uuid.UUID) -> int:
        """Delete all chunks belonging to a document version. Return count of deleted rows."""
        try:
            stmt = delete(DocumentChunk).where(DocumentChunk.document_version_id == version_id)
            result = await self.session.execute(stmt)
            await self.session.flush()
            rowcount = getattr(result, "rowcount", 0)
            return int(rowcount if rowcount is not None else 0)
        except SQLAlchemyError as exc:
            await self.session.rollback()
            logger.error(f"Error deleting chunks for version {version_id}: {exc}")
            raise DatabaseError(f"Database error deleting chunks: {exc}") from exc

    async def update_embedding_reference(self, chunk_id: uuid.UUID, embedding_id: str) -> None:
        """Associate a chunk with an external vector store identifier."""
        chunk = await self.get_by_id(chunk_id)
        if not chunk:
            raise EntityNotFoundError(entity_name="DocumentChunk", entity_id=str(chunk_id))

        try:
            chunk.embedding_id = embedding_id
            await self.session.flush()
        except SQLAlchemyError as exc:
            await self.session.rollback()
            logger.error(f"Error updating embedding reference for chunk {chunk_id}: {exc}")
            raise DatabaseError(f"Database error updating embedding reference: {exc}") from exc
