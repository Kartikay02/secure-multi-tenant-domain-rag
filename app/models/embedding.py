"""ChunkEmbedding SQLAlchemy declarative model for pgvector indexing."""

import uuid
from typing import TYPE_CHECKING, Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Index, Integer, String, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, JSONType, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.chunk import DocumentChunk
    from app.models.document import Document


class ChunkEmbedding(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Stores dense vector embeddings with foreign key links and HNSW indexing."""

    __tablename__ = "chunk_embeddings"

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("document_chunks.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)

    # Relationships
    chunk: Mapped["DocumentChunk"] = relationship(back_populates="embedding_entry")
    document: Mapped["Document"] = relationship()

    __table_args__ = (
        UniqueConstraint("chunk_id", name="uq_chunk_embedding_chunk_id"),
        Index("ix_chunk_embeddings_chunk_id", "chunk_id"),
        Index("ix_chunk_embeddings_document_id", "document_id"),
        Index("ix_chunk_embeddings_model", "model"),
        Index(
            "ix_chunk_embeddings_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<ChunkEmbedding(id={self.id}, chunk_id={self.chunk_id}, "
            f"model='{self.model}', dim={self.dimension})>"
        )
