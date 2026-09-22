"""IngestionJob repository protocol interface."""

import uuid
from typing import Any, Protocol

from app.models.job import IngestionJob
from app.repositories.interfaces.base import BaseRepositoryProtocol


class IngestionJobRepositoryProtocol(BaseRepositoryProtocol[IngestionJob], Protocol):
    """Data access operations specific to asynchronous IngestionJobs."""

    async def create_job(
        self, document_id: uuid.UUID, document_version_id: uuid.UUID | None = None
    ) -> IngestionJob:
        """Initialize and persist a new ingestion job record."""
        ...

    async def update_progress(
        self,
        job_id: uuid.UUID,
        status: str,
        stage_name: str | None = None,
        error_message: str | None = None,
        stats: dict[str, Any] | None = None,
    ) -> IngestionJob:
        """Update job status, record completed stages, and register execution metrics."""
        ...

    async def get_jobs_by_document(self, document_id: uuid.UUID) -> list[IngestionJob]:
        """Fetch all ingestion jobs associated with a given document."""
        ...
