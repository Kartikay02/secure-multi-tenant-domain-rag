"""DocumentChunk repository protocol interface."""

import uuid
from typing import Protocol

from app.models.chunk import DocumentChunk
from app.repositories.interfaces.base import BaseRepositoryProtocol


class DocumentChunkRepositoryProtocol(BaseRepositoryProtocol[DocumentChunk], Protocol):
    """Data access operations specific to DocumentChunk entities."""

    async def bulk_create_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Atomically persist a batch of document chunks."""
        ...

    async def get_chunks_by_version(
        self, version_id: uuid.UUID, include_content: bool = True
    ) -> list[DocumentChunk]:
        """Retrieve all chunks for a document version.

        When include_content=False, defers loading chunk text payload to save I/O and memory.
        """
        ...

    async def count_by_version(self, version_id: uuid.UUID) -> int:
        """Count total chunks belonging to a document version."""
        ...

    async def delete_chunks_by_version(self, version_id: uuid.UUID) -> int:
        """Delete all chunks belonging to a document version. Return count of deleted rows."""
        ...

    async def update_embedding_reference(self, chunk_id: uuid.UUID, embedding_id: str) -> None:
        """Associate a chunk with an external vector store or embedding identifier."""
        ...
