"""Application service for chunking document versions and persisting chunks."""

import time
import uuid

from app.core.exceptions import EntityNotFoundError, IngestionError
from app.core.logging import get_logger
from app.models.chunk import DocumentChunk
from app.rag.chunking.interfaces import ChunkerProtocol
from app.rag.chunking.recursive import RecursiveTokenChunker
from app.repositories.interfaces.chunk import DocumentChunkRepositoryProtocol
from app.repositories.interfaces.document import DocumentRepositoryProtocol
from app.repositories.interfaces.job import IngestionJobRepositoryProtocol
from app.repositories.interfaces.version import DocumentVersionRepositoryProtocol

logger = get_logger("app.service.chunking")


class ChunkingService:
    """Orchestrates document chunking and atomic persistence for document revisions."""

    def __init__(
        self,
        document_repo: DocumentRepositoryProtocol,
        version_repo: DocumentVersionRepositoryProtocol,
        chunk_repo: DocumentChunkRepositoryProtocol,
        job_repo: IngestionJobRepositoryProtocol,
        chunker: ChunkerProtocol | None = None,
    ) -> None:
        self.document_repo = document_repo
        self.version_repo = version_repo
        self.chunk_repo = chunk_repo
        self.job_repo = job_repo
        self.chunker = chunker or RecursiveTokenChunker()

    async def chunk_document_version(
        self,
        document_id: uuid.UUID,
        version_id: uuid.UUID,
    ) -> list[DocumentChunk]:
        """Split a document version into chunks and persist them idempotently."""
        start_time = time.perf_counter()
        logger.info(f"Starting chunking for document {document_id}, version {version_id}")

        document = await self.document_repo.get_by_id(document_id)
        if not document:
            raise EntityNotFoundError(entity_name="Document", entity_id=str(document_id))

        version = await self.version_repo.get_by_id(version_id)
        if not version:
            raise EntityNotFoundError(entity_name="DocumentVersion", entity_id=str(version_id))

        if version.status == "FAILED":
            raise IngestionError(
                f"Cannot chunk document version {version_id}: Current status is FAILED."
            )

        # Retrieve parsed sections from metadata
        meta = version.metadata_json or {}
        sections: list[dict[str, str]] = meta.get("sections", [])

        # If sections list is empty, construct a single fallback section from raw content
        if not sections and "raw_text" in meta:
            sections = [{"title": document.name, "content": meta["raw_text"]}]

        if not sections:
            logger.warning(
                f"No text sections found in metadata for version {version_id}. Yielding 0 chunks."
            )
            await self.version_repo.update_status(
                version_id=version_id, status="CHUNKED", total_chunks=0
            )
            return []

        tenant_id = (
            (document.metadata_json or {}).get("tenant_id") or meta.get("tenant_id") or "default"
        )
        base_metadata = {
            "document_id": str(document_id),
            "document_version_id": str(version_id),
            "document_name": document.name,
            "document_type": document.document_type,
            "source": document.source,
            "tenant_id": str(tenant_id),
        }

        # 1. Execute semantic / recursive chunking
        chunk_payloads = self.chunker.split_sections(
            sections=sections,
            base_metadata=base_metadata,
        )

        db_chunks = [
            DocumentChunk(
                document_version_id=version_id,
                chunk_index=payload.chunk_index,
                content=payload.content,
                token_count=payload.token_count,
                char_count=payload.char_count,
                metadata_json=payload.metadata,
            )
            for payload in chunk_payloads
        ]

        # 2. Idempotency guarantee: delete existing chunks for this version prior to insert
        deleted_count = await self.chunk_repo.delete_chunks_by_version(version_id)
        if deleted_count > 0:
            logger.info(
                f"Removed {deleted_count} stale chunks for version {version_id} prior to re-chunking."
            )

        # 3. Persist new chunks in batch
        persisted_chunks = await self.chunk_repo.bulk_create_chunks(db_chunks)

        # 4. Update DocumentVersion state
        await self.version_repo.update_status(
            version_id=version_id,
            status="CHUNKED",
            total_chunks=len(persisted_chunks),
        )

        # 5. Update IngestionJob telemetry
        jobs = await self.job_repo.get_jobs_by_document(document_id)
        if jobs:
            latest_job = jobs[0]
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            await self.job_repo.update_progress(
                job_id=latest_job.id,
                status="CHUNKED",
                stage_name="chunking",
                stats={"chunk_count": len(persisted_chunks), "chunking_ms": elapsed_ms},
            )

        logger.info(
            f"Successfully generated and persisted {len(persisted_chunks)} chunks "
            f"for version {version_id} in {round((time.perf_counter() - start_time) * 1000, 2)}ms"
        )
        return persisted_chunks
