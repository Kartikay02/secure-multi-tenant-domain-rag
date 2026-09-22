"""Add chunk_embeddings table with pgvector HNSW indexing.

Revision ID: 0002_add_chunk_embeddings
Revises: 0001_initial_schema
Create Date: 2026-09-08 19:45:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_add_chunk_embeddings"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "chunk_embeddings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("chunk_id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column(
            "metadata_json",
            postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["document_chunks.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("chunk_id", name="uq_chunk_embedding_chunk_id"),
    )

    op.create_index("ix_chunk_embeddings_chunk_id", "chunk_embeddings", ["chunk_id"], unique=False)
    op.create_index(
        "ix_chunk_embeddings_document_id", "chunk_embeddings", ["document_id"], unique=False
    )
    op.create_index("ix_chunk_embeddings_model", "chunk_embeddings", ["model"], unique=False)

    if bind.dialect.name == "postgresql":
        op.create_index(
            "ix_chunk_embeddings_hnsw",
            "chunk_embeddings",
            ["embedding"],
            unique=False,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        )
        op.create_index(
            "ix_chunk_embeddings_meta_gin",
            "chunk_embeddings",
            ["metadata_json"],
            unique=False,
            postgresql_using="gin",
        )


def downgrade() -> None:
    op.drop_table("chunk_embeddings")
