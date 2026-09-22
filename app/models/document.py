"""Document and DocumentMetadata SQLAlchemy declarative models."""

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import ForeignKey, Index, String, Text, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, JSONType, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.job import IngestionJob
    from app.models.version import DocumentVersion


class Document(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Core domain entity representing a managed document in the RAG system."""

    __tablename__ = "documents"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    document_type: Mapped[str] = mapped_column(String(50), nullable=False)
    source: Mapped[str] = mapped_column(String(1024), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)

    # Relationships
    versions: Mapped[list["DocumentVersion"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentVersion.version_number.desc()",
        lazy="selectin",
    )
    metadata_entries: Mapped[list["DocumentMetadata"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    ingestion_jobs: Mapped[list["IngestionJob"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        Index("ix_documents_name", "name"),
        Index("ix_documents_source", "source"),
        Index("ix_documents_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, name='{self.name}', type='{self.document_type}')>"


class DocumentMetadata(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Typed key-value metadata attribute associated with a Document."""

    __tablename__ = "document_metadata"

    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    key: Mapped[str] = mapped_column(String(128), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    value_type: Mapped[str] = mapped_column(String(32), default="string", nullable=False)

    # Relationship
    document: Mapped["Document"] = relationship(back_populates="metadata_entries")

    __table_args__ = (
        UniqueConstraint("document_id", "key", name="uq_document_metadata_key"),
        Index("ix_document_metadata_key_val", "key", "value"),
    )

    def __repr__(self) -> str:
        return f"<DocumentMetadata(doc_id={self.document_id}, key='{self.key}', value='{self.value[:20]}')>"
