"""IngestionJob repository implementation."""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import DatabaseError, EntityNotFoundError
from app.core.logging import get_logger
from app.models.job import IngestionJob
from app.repositories.base import BaseSQLAlchemyRepository
from app.repositories.interfaces.job import IngestionJobRepositoryProtocol

logger = get_logger("app.repository.job")


class SQLAlchemyIngestionJobRepository(
    BaseSQLAlchemyRepository[IngestionJob], IngestionJobRepositoryProtocol
):
    """PostgreSQL data access implementation for IngestionJobs."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session=session, entity_class=IngestionJob)

    async def create_job(
        self, document_id: uuid.UUID, document_version_id: uuid.UUID | None = None
    ) -> IngestionJob:
        """Initialize and persist a new ingestion job record."""
        job = IngestionJob(
            document_id=document_id,
            document_version_id=document_version_id,
            status="QUEUED",
            stages_completed={},
            stats={},
            started_at=datetime.now(UTC),
        )
        return await self.create(job)

    async def update_progress(
        self,
        job_id: uuid.UUID,
        status: str,
        stage_name: str | None = None,
        error_message: str | None = None,
        stats: dict[str, Any] | None = None,
    ) -> IngestionJob:
        """Update job status, record completed stages, and register execution metrics."""
        job = await self.get_by_id(job_id)
        if not job:
            raise EntityNotFoundError(entity_name="IngestionJob", entity_id=str(job_id))

        try:
            job.status = status
            if stage_name:
                updated_stages = dict(job.stages_completed)
                updated_stages[stage_name] = datetime.now(UTC).isoformat()
                job.stages_completed = updated_stages

            if error_message:
                job.error_message = error_message

            if stats:
                updated_stats = dict(job.stats)
                updated_stats.update(stats)
                job.stats = updated_stats

            if status in ("COMPLETED", "FAILED"):
                job.completed_at = datetime.now(UTC)

            await self.session.flush()
            return job
        except SQLAlchemyError as exc:
            await self.session.rollback()
            logger.error(f"Error updating ingestion job {job_id}: {exc}")
            raise DatabaseError(f"Database error updating ingestion job: {exc}") from exc

    async def get_jobs_by_document(self, document_id: uuid.UUID) -> list[IngestionJob]:
        """Fetch all ingestion jobs associated with a given document."""
        try:
            query = (
                select(IngestionJob)
                .where(IngestionJob.document_id == document_id)
                .order_by(IngestionJob.created_at.desc())
            )
            result = await self.session.execute(query)
            return list(result.scalars().all())
        except SQLAlchemyError as exc:
            logger.error(f"Error fetching jobs for document {document_id}: {exc}")
            raise DatabaseError(f"Database error fetching jobs for document: {exc}") from exc
