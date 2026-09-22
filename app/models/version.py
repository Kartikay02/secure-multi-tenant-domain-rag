"""DocumentVersion SQLAlchemy declarative model."""

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import BigInteger, ForeignKey, Index, Integer, String, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, JSONType, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.chunk import DocumentChunk
    from app.models.document import Document
    from app.models.job import IngestionJob


class DocumentVersion(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Specific revision of a Document with associated chunks and processing status."""

    __tablename__ = "document_versions"

    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    version_number: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="PENDING", nullable=False)
    total_chunks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)

    # Relationships
    document: Mapped["Document"] = relationship(back_populates="versions")
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="version",
        cascade="all, delete-orphan",
        order_by="DocumentChunk.chunk_index.asc()",
    )
    ingestion_jobs: Mapped[list["IngestionJob"]] = relationship(
        back_populates="document_version",
    )

    __table_args__ = (
        UniqueConstraint("document_id", "version_number", name="uq_doc_version_number"),
        Index("ix_doc_version_hash", "content_hash"),
        Index("ix_doc_version_status", "status"),
        Index("ix_doc_version_doc_id", "document_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<DocumentVersion(id={self.id}, doc_id={self.document_id}, "
            f"v={self.version_number}, status='{self.status}')>"
        )
