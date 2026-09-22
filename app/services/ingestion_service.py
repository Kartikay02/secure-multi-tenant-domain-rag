"""Application service orchestrating document upload, validation, storage, and parsing."""

import time
import uuid
from typing import Any

from app.core.exceptions import AppException, EntityNotFoundError
from app.core.logging import get_logger
from app.models.document import Document
from app.models.version import DocumentVersion
from app.rag.ingestion.normalizer import TextNormalizer
from app.rag.ingestion.parsers.factory import ParserFactory, default_parser_factory
from app.rag.ingestion.validator import FileValidator
from app.rag.storage.interfaces import FileStorageProtocol
from app.repositories.interfaces.document import DocumentRepositoryProtocol
from app.repositories.interfaces.job import IngestionJobRepositoryProtocol
from app.repositories.interfaces.version import DocumentVersionRepositoryProtocol
from app.schemas.ingestion import IngestionResponse

logger = get_logger("app.service.ingestion")


class IngestionService:
    """Coordinates validation, file storage, parsing, normalization, and persistence."""

    def __init__(
        self,
        document_repo: DocumentRepositoryProtocol,
        version_repo: DocumentVersionRepositoryProtocol,
        job_repo: IngestionJobRepositoryProtocol,
        storage_service: FileStorageProtocol,
        validator: FileValidator | None = None,
        parser_factory: ParserFactory | None = None,
    ) -> None:
        self.document_repo = document_repo
        self.version_repo = version_repo
        self.job_repo = job_repo
        self.storage_service = storage_service
        self.validator = validator or FileValidator()
        self.parser_factory = parser_factory or default_parser_factory

    async def ingest_file(
        self,
        file_bytes: bytes,
        filename: str,
        custom_metadata: dict[str, Any] | None = None,
    ) -> IngestionResponse:
        """Execute end-to-end ingestion pipeline on uploaded file bytes."""
        start_time = time.perf_counter()
        logger.info(f"Starting ingestion for file: '{filename}' ({len(file_bytes)} bytes)")

        # 1. Validation & Cryptographic Hashing
        validation = self.validator.validate(file_bytes=file_bytes, raw_filename=filename)

        tenant_id = (custom_metadata or {}).get("tenant_id", "default")

        # 2. Duplicate Detection (Idempotency)
        # SEC-09: Scope hash lookup to tenant to prevent cross-tenant enumeration leaks
        existing_version = await self.version_repo.get_by_content_hash(
            validation.content_hash, tenant_id=tenant_id
        )
        if existing_version is not None:
            logger.info(
                f"Duplicate document detected for tenant '{tenant_id}' (hash={validation.content_hash[:12]}...). Skipping re-indexing."
            )
            parent_doc = await self.document_repo.get_by_id(existing_version.document_id)
            if not parent_doc:
                raise EntityNotFoundError(
                    entity_name="Document", entity_id=str(existing_version.document_id)
                )

            # Retrieve active or latest job
            jobs = await self.job_repo.get_jobs_by_document(parent_doc.id)
            latest_job_id = jobs[0].id if jobs else uuid.uuid4()

            meta = existing_version.metadata_json
            return IngestionResponse(
                document_id=parent_doc.id,
                document_version_id=existing_version.id,
                job_id=latest_job_id,
                filename=validation.filename,
                content_hash=validation.content_hash,
                size_bytes=validation.size_bytes,
                status=existing_version.status,
                is_duplicate=True,
                extracted_title=meta.get("title", parent_doc.name),
                extracted_author=meta.get("author"),
                word_count=meta.get("word_count", 0),
                char_count=meta.get("char_count", 0),
                estimated_tokens=meta.get("estimated_tokens", 0),
                storage_uri=parent_doc.source,
                message="Duplicate file detected with matching content hash; skipped re-processing.",
            )

        # 3. Persistent File Storage
        storage_uri = await self.storage_service.save(
            file_bytes=file_bytes,
            filename=validation.filename,
            content_type=validation.detected_mime_type,
        )

        # 4. Create Document and Initial Version with compensating rollback
        doc_type = validation.extension.lstrip(".")
        document = Document(
            name=validation.filename,
            document_type=doc_type,
            source=storage_uri,
            metadata_json=custom_metadata or {},
        )
        version = DocumentVersion(
            document_id=uuid.uuid4(),  # updated atomically by repo
            version_number=1,
            content_hash=validation.content_hash,
            size_bytes=validation.size_bytes,
            status="PROCESSING",
            metadata_json={"storage_uri": storage_uri, "tenant_id": tenant_id},
        )

        try:
            persisted_doc = await self.document_repo.create_with_version(document, version)
        except Exception:
            # Compensating action: remove saved file if persistence failed
            try:
                await self.storage_service.delete(storage_uri)
            except Exception:
                pass
            raise

        # 5. Create Ingestion Job Record
        job = await self.job_repo.create_job(
            document_id=persisted_doc.id,
            document_version_id=version.id,
        )
        await self.job_repo.update_progress(
            job_id=job.id,
            status="PROCESSING",
            stage_name="storage",
        )

        # 6. Format-Specific Parsing
        try:
            parser = self.parser_factory.get_parser(validation.filename)
            parsed_doc = await parser.parse(file_bytes=file_bytes, filename=validation.filename)
            await self.job_repo.update_progress(
                job_id=job.id,
                status="PROCESSING",
                stage_name="parsing",
            )
        except Exception as exc:
            err_msg = str(exc)
            logger.error(f"Parsing failed for '{validation.filename}': {err_msg}", exc_info=True)
            await self.job_repo.update_progress(
                job_id=job.id,
                status="FAILED",
                error_message=err_msg,
            )
            await self.version_repo.update_status(version_id=version.id, status="FAILED")
            raise exc

        # 7. Text Normalization
        try:
            normalized = TextNormalizer.normalize(parsed_doc.content)

            # Assemble complete version metadata
            combined_meta: dict[str, Any] = {
                "storage_uri": storage_uri,
                "title": parsed_doc.title or validation.filename,
                "author": parsed_doc.author,
                "char_count": normalized.char_count,
                "word_count": normalized.word_count,
                "estimated_tokens": normalized.estimated_token_count,
                "section_count": len(parsed_doc.sections),
                "sections": parsed_doc.sections,
                **parsed_doc.metadata,
            }
            if custom_metadata:
                combined_meta.update(custom_metadata)

            version.metadata_json = combined_meta
            await self.version_repo.update_status(version_id=version.id, status="PARSED")

            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            stats = {
                "char_count": normalized.char_count,
                "word_count": normalized.word_count,
                "estimated_tokens": normalized.estimated_token_count,
                "section_count": len(parsed_doc.sections),
                "duration_ms": elapsed_ms,
            }

            await self.job_repo.update_progress(
                job_id=job.id,
                status="COMPLETED",
                stage_name="normalization",
                stats=stats,
            )

            logger.info(
                f"Ingestion completed for '{validation.filename}' in {elapsed_ms}ms "
                f"({normalized.word_count} words, ~{normalized.estimated_token_count} tokens)"
            )

            return IngestionResponse(
                document_id=persisted_doc.id,
                document_version_id=version.id,
                job_id=job.id,
                filename=validation.filename,
                content_hash=validation.content_hash,
                size_bytes=validation.size_bytes,
                status="PARSED",
                is_duplicate=False,
                extracted_title=parsed_doc.title,
                extracted_author=parsed_doc.author,
                word_count=normalized.word_count,
                char_count=normalized.char_count,
                estimated_tokens=normalized.estimated_token_count,
                storage_uri=storage_uri,
                message="Document successfully validated, stored, parsed, and normalized.",
            )

        except AppException:
            raise
        except Exception as exc:
            err_msg = f"Unexpected error during normalization: {exc}"
            logger.error(err_msg, exc_info=True)
            await self.job_repo.update_progress(
                job_id=job.id,
                status="FAILED",
                error_message=err_msg,
            )
            await self.version_repo.update_status(version_id=version.id, status="FAILED")
            raise exc
