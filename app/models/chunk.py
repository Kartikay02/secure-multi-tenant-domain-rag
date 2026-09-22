"""DocumentChunk SQLAlchemy declarative model."""

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import ForeignKey, Index, Integer, String, Text, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, JSONType, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.embedding import ChunkEmbedding
    from app.models.version import DocumentVersion


class DocumentChunk(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Segmented chunk of a DocumentVersion with token metrics and embedding link."""

    __tablename__ = "document_chunks"

    document_version_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("document_versions.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    embedding_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)

    # Relationships
    version: Mapped["DocumentVersion"] = relationship(back_populates="chunks")
    embedding_entry: Mapped["ChunkEmbedding | None"] = relationship(
        back_populates="chunk",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        UniqueConstraint("document_version_id", "chunk_index", name="uq_chunk_version_index"),
        Index("ix_chunk_version_idx", "document_version_id", "chunk_index"),
        Index("ix_chunk_embedding_id", "embedding_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<DocumentChunk(id={self.id}, version_id={self.document_version_id}, "
            f"idx={self.chunk_index}, tokens={self.token_count})>"
        )
