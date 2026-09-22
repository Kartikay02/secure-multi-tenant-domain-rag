"""Service orchestrating text embedding generation and vector persistence."""

import uuid
from typing import Any

from app.core.exceptions import EntityNotFoundError
from app.core.logging import get_logger
from app.rag.embeddings.interfaces import EmbeddingProtocol
from app.rag.embeddings.vector_interfaces import VectorRecord
from app.rag.embeddings.vector_mock import InMemoryVectorStore
from app.repositories.interfaces.chunk import DocumentChunkRepositoryProtocol
from app.repositories.interfaces.document import DocumentRepositoryProtocol
from app.repositories.interfaces.job import IngestionJobRepositoryProtocol
from app.repositories.interfaces.version import DocumentVersionRepositoryProtocol

logger = get_logger("app.services.embedding")


class EmbeddingService:
    """Orchestrates embedding generation, vector store persistence, and version status updates."""

    def __init__(
        self,
        document_repo: DocumentRepositoryProtocol,
        version_repo: DocumentVersionRepositoryProtocol,
        chunk_repo: DocumentChunkRepositoryProtocol,
        job_repo: IngestionJobRepositoryProtocol,
        embedding_provider: EmbeddingProtocol,
        vector_store: Any = None,
    ) -> None:
        self.document_repo = document_repo
        self.version_repo = version_repo
        self.chunk_repo = chunk_repo
        self.job_repo = job_repo
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store or InMemoryVectorStore()

    async def embed_document_version(
        self,
        document_id: uuid.UUID,
        version_id: uuid.UUID,
    ) -> list[VectorRecord]:
        """Generate and persist vector embeddings for all chunks in a DocumentVersion.

        Args:
            document_id: Target Document UUID.
            version_id: Target DocumentVersion UUID.

        Returns:
            List of generated and persisted VectorRecord instances.

        Raises:
            EntityNotFoundError: If the document or version does not exist.
            EmbeddingError: If vector generation or persistence fails.
        """
        logger.info(
            f"Starting embedding generation for document {document_id}, version {version_id}",
            extra={"document_id": str(document_id), "version_id": str(version_id)},
        )

        doc = await self.document_repo.get_by_id(document_id)
        if not doc:
            raise EntityNotFoundError("Document", str(document_id))

        version = await self.version_repo.get_by_id(version_id)
        if not version:
            raise EntityNotFoundError("DocumentVersion", str(version_id))

        chunks = await self.chunk_repo.get_chunks_by_version(version_id)
        if not chunks:
            logger.warning(f"No chunks found for document version {version_id} to embed.")
            return []

        # Update IngestionJob telemetry if one exists
        jobs = await self.job_repo.get_jobs_by_document(document_id)
        active_job = jobs[0] if jobs else None
        if active_job:
            await self.job_repo.update_progress(
                job_id=active_job.id,
                status="PROCESSING",
                stage_name="embedding_started",
            )

        try:
            # 1. Extract texts to embed
            texts = [c.content for c in chunks]

            # 2. Generate vector embeddings in batches
            vectors = await self.embedding_provider.embed_documents(texts)

            # 3. Construct VectorRecords
            records: list[VectorRecord] = []
            for chunk, vec in zip(chunks, vectors, strict=True):
                vector_id = str(chunk.id)
                chunk_meta = chunk.metadata_json.copy() if chunk.metadata_json else {}
                chunk_meta.update(
                    {
                        "chunk_index": chunk.chunk_index,
                        "token_count": chunk.token_count,
                        "char_count": chunk.char_count,
                        "document_name": doc.name,
                        "document_type": doc.document_type,
                        "content": chunk.content,
                        "text": chunk.content,
                    }
                )
                rec = VectorRecord(
                    id=vector_id,
                    vector=vec,
                    document_id=str(document_id),
                    version_id=str(version_id),
                    metadata=chunk_meta,
                )
                records.append(rec)

            # 4. Upsert into vector store
            if hasattr(self.vector_store, "upsert_vectors"):
                await self.vector_store.upsert_vectors(records)
            else:
                await self.vector_store.upsert(records)

            # 5. Update embedding references on each chunk
            for chunk, rec in zip(chunks, records, strict=True):
                await self.chunk_repo.update_embedding_reference(chunk.id, rec.id)

            # 6. Advance version status
            await self.version_repo.update_status(version_id=version_id, status="EMBEDDED")

            # 7. Complete job stage
            if active_job:
                await self.job_repo.update_progress(
                    job_id=active_job.id,
                    status="COMPLETED",
                    stage_name="embedding",
                    stats={
                        "total_vectors": len(records),
                        "dimension": self.embedding_provider.dimension,
                    },
                )

            logger.info(
                f"Successfully embedded {len(records)} chunks for version {version_id}",
                extra={
                    "version_id": str(version_id),
                    "vector_count": len(records),
                    "dimension": self.embedding_provider.dimension,
                },
            )
            return records

        except Exception as exc:
            logger.error(
                f"Embedding pipeline failed for version {version_id}: {exc}",
                extra={"version_id": str(version_id), "error": str(exc)},
            )
            if active_job:
                await self.job_repo.update_progress(
                    job_id=active_job.id,
                    status="FAILED",
                    error_message=str(exc),
                )
            raise
