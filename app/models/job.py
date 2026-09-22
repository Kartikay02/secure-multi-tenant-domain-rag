"""IngestionJob SQLAlchemy declarative model."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, JSONType, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.document import Document
    from app.models.version import DocumentVersion


class IngestionJob(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Tracks asynchronous document processing, parsing, chunking, and embedding stages."""

    __tablename__ = "ingestion_jobs"

    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_version_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("document_versions.id", ondelete="SET NULL"),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(String(32), default="QUEUED", nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    stages_completed: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    stats: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    document: Mapped["Document"] = relationship(back_populates="ingestion_jobs")
    document_version: Mapped["DocumentVersion | None"] = relationship(
        back_populates="ingestion_jobs"
    )

    __table_args__ = (
        Index("ix_ingestion_jobs_doc_status", "document_id", "status"),
        Index("ix_ingestion_jobs_status_created", "status", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<IngestionJob(id={self.id}, doc_id={self.document_id}, status='{self.status}')>"
