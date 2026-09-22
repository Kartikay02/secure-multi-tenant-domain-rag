"""Document management, ingestion, processing, and listing endpoints."""

import json
import time
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, Query, Response, UploadFile, status

from app.api.deps import (
    check_rate_limit,
    get_authorization_policy,
    get_chunk_repository,
    get_chunking_service,
    get_document_repository,
    get_embedding_service,
    get_ingestion_service,
    get_job_repository,
    get_security_context,
    get_storage_service,
    get_version_repository,
    verify_api_key,
)
from app.core.config import get_settings
from app.core.exceptions import (
    EntityNotFoundError,
    FileSizeLimitExceededError,
    FileValidationError,
    ForbiddenError,
)
from app.core.logging import get_logger
from app.rag.storage.interfaces import FileStorageProtocol
from app.repositories.interfaces.chunk import DocumentChunkRepositoryProtocol
from app.repositories.interfaces.document import DocumentRepositoryProtocol
from app.repositories.interfaces.job import IngestionJobRepositoryProtocol
from app.repositories.interfaces.version import DocumentVersionRepositoryProtocol
from app.schemas.common import PaginatedResponse
from app.schemas.ingestion import (
    ChunkResponse,
    DocumentDetailResponse,
    DocumentListItemResponse,
    DocumentMetadataResponse,
    DocumentProcessResponse,
    DocumentVersionResponse,
    IngestionJobResponse,
    IngestionResponse,
)
from app.security.authorization import (
    AuthorizationPolicyProtocol,
    DocumentAction,
    SecurityContext,
)
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.ingestion_service import IngestionService

logger = get_logger("app.api.documents")

router = APIRouter()

IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]
ChunkingServiceDep = Annotated[ChunkingService, Depends(get_chunking_service)]
EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]
DocumentRepoDep = Annotated[DocumentRepositoryProtocol, Depends(get_document_repository)]
VersionRepoDep = Annotated[DocumentVersionRepositoryProtocol, Depends(get_version_repository)]
JobRepoDep = Annotated[IngestionJobRepositoryProtocol, Depends(get_job_repository)]
SecurityContextDep = Annotated[SecurityContext, Depends(get_security_context)]
AuthzPolicyDep = Annotated[AuthorizationPolicyProtocol, Depends(get_authorization_policy)]
ChunkRepoDep = Annotated[DocumentChunkRepositoryProtocol, Depends(get_chunk_repository)]
StorageDep = Annotated[FileStorageProtocol, Depends(get_storage_service)]


async def _handle_upload(
    response: Response,
    file: UploadFile,
    service: IngestionService,
    sec_ctx: SecurityContext,
    metadata: str | None = None,
) -> IngestionResponse:
    """Core upload handler validating file and invoking IngestionService."""
    settings = get_settings()
    max_bytes = settings.security.max_file_size_mb * 1024 * 1024

    # SEC-10: Bounded stream read enforcing upload size threshold without full memory buffering
    CHUNK_SIZE = 64 * 1024
    chunks: list[bytes] = []
    total_size = 0

    try:
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > max_bytes:
                logger.warning(
                    f"Upload rejected: payload size {total_size} exceeds configured limit of {max_bytes} bytes."
                )
                raise FileSizeLimitExceededError(size_bytes=total_size, max_bytes=max_bytes)
            chunks.append(chunk)
    finally:
        await file.close()

    file_bytes = b"".join(chunks)
    filename = file.filename or "unknown_document"

    custom_meta: dict[str, Any] = {}
    if metadata:
        try:
            custom_meta = json.loads(metadata)
            if not isinstance(custom_meta, dict):
                raise ValueError("Metadata must be a valid JSON object.")
        except Exception as exc:
            raise FileValidationError(f"Invalid metadata JSON: {exc}") from exc

    # SEC-02 / SEC-48: Enforce strict multi-tenant boundary
    if sec_ctx.is_authenticated and not sec_ctx.is_admin():
        target_tenant = custom_meta.get("tenant_id")
        if target_tenant and target_tenant != sec_ctx.tenant_id:
            raise ForbiddenError(
                f"Cannot upload document for tenant '{target_tenant}'. Caller is restricted to tenant '{sec_ctx.tenant_id}'."
            )
        custom_meta["tenant_id"] = sec_ctx.tenant_id
    elif sec_ctx.is_admin():
        custom_meta["tenant_id"] = custom_meta.get("tenant_id") or sec_ctx.tenant_id
    else:
        custom_meta["tenant_id"] = custom_meta.get("tenant_id") or sec_ctx.tenant_id

    result = await service.ingest_file(
        file_bytes=file_bytes,
        filename=filename,
        custom_metadata=custom_meta,
    )

    if result.is_duplicate:
        response.status_code = status.HTTP_200_OK

    return result


@router.post(
    "",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest document",
    description="Upload a document (.pdf, .docx, .txt, .md) to store, parse, normalize, and register for chunking.",
    dependencies=[Depends(check_rate_limit), Depends(verify_api_key)],
)
async def upload_document_root(
    response: Response,
    file: Annotated[
        UploadFile,
        File(description="Document file to upload (.pdf, .docx, .txt, .md)"),
    ],
    service: IngestionServiceDep,
    sec_ctx: SecurityContextDep,
    metadata: Annotated[
        str | None,
        Form(description="Optional JSON-encoded string containing custom metadata"),
    ] = None,
) -> IngestionResponse:
    """Primary document upload endpoint (POST /documents)."""
    return await _handle_upload(response, file, service, sec_ctx, metadata)


@router.post(
    "/upload",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest document (alias)",
    description="Alias for POST /documents for backwards compatibility.",
    dependencies=[Depends(check_rate_limit), Depends(verify_api_key)],
    include_in_schema=True,
)
async def upload_document_alias(
    response: Response,
    file: Annotated[
        UploadFile,
        File(description="Document file to upload (.pdf, .docx, .txt, .md)"),
    ],
    service: IngestionServiceDep,
    sec_ctx: SecurityContextDep,
    metadata: Annotated[
        str | None,
        Form(description="Optional JSON-encoded string containing custom metadata"),
    ] = None,
) -> IngestionResponse:
    """Backwards-compatible upload alias (POST /documents/upload)."""
    return await _handle_upload(response, file, service, sec_ctx, metadata)


@router.get(
    "",
    response_model=PaginatedResponse[DocumentListItemResponse],
    status_code=status.HTTP_200_OK,
    summary="List documents with pagination",
    description="Retrieve a paginated list of all stored documents.",
    dependencies=[Depends(check_rate_limit), Depends(verify_api_key)],
)
async def list_documents(
    doc_repo: DocumentRepoDep,
    sec_ctx: SecurityContextDep,
    page: Annotated[int, Query(ge=1, description="Page number (1-based)")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 10,
    tenant_id: Annotated[
        str | None, Query(description="Target tenant filter for admin callers")
    ] = None,
) -> PaginatedResponse[DocumentListItemResponse]:
    """List documents with pagination and strict tenant scoping (SEC-03)."""
    offset = (page - 1) * page_size

    if sec_ctx.is_admin():
        target = tenant_id or (sec_ctx.tenant_id if sec_ctx.tenant_id != "default" else None)
        if target:
            total = await doc_repo.count_by_tenant(target)
            docs = await doc_repo.list_by_tenant(target, offset=offset, limit=page_size)
        else:
            total = await doc_repo.count()
            docs = await doc_repo.list(offset=offset, limit=page_size)
    else:
        target = sec_ctx.tenant_id
        total = await doc_repo.count_by_tenant(target)
        docs = await doc_repo.list_by_tenant(target, offset=offset, limit=page_size)

    items = [
        DocumentListItemResponse(
            id=d.id,
            name=d.name,
            document_type=d.document_type,
            source=d.source,
            description=d.description,
            created_at=d.created_at,
            updated_at=d.updated_at,
        )
        for d in docs
    ]
    total_pages = (total + page_size - 1) // page_size if total > 0 else 1

    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get document details",
    description="Retrieve a document with all its revisions and metadata entries.",
    dependencies=[Depends(check_rate_limit), Depends(verify_api_key)],
)
async def get_document(
    document_id: uuid.UUID,
    doc_repo: DocumentRepoDep,
    sec_ctx: SecurityContextDep,
    authz_policy: AuthzPolicyDep,
) -> DocumentDetailResponse:
    """Fetch complete document metadata and version history with tenant authorization."""
    doc = await doc_repo.get_with_versions(document_id)
    if not doc:
        raise EntityNotFoundError(entity_name="Document", entity_id=str(document_id))

    authz_policy.authorize_document(sec_ctx, doc.metadata_json, DocumentAction.READ)

    return DocumentDetailResponse(
        id=doc.id,
        name=doc.name,
        document_type=doc.document_type,
        source=doc.source,
        description=doc.description,
        metadata_json=doc.metadata_json,
        versions=[
            DocumentVersionResponse(
                id=v.id,
                version_number=v.version_number,
                content_hash=v.content_hash,
                size_bytes=v.size_bytes,
                status=v.status,
                total_chunks=v.total_chunks,
                metadata_json=v.metadata_json,
                created_at=v.created_at,
            )
            for v in doc.versions
        ],
        metadata_entries=[
            DocumentMetadataResponse(
                key=m.key,
                value=m.value,
                value_type=m.value_type,
            )
            for m in doc.metadata_entries
        ],
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


@router.post(
    "/{document_id}/process",
    response_model=DocumentProcessResponse,
    status_code=status.HTTP_200_OK,
    summary="Process document (chunking and embedding)",
    description="Executes semantic chunking and embedding generation for the latest version of a document.",
    dependencies=[Depends(check_rate_limit), Depends(verify_api_key)],
)
async def process_document(
    document_id: uuid.UUID,
    doc_repo: DocumentRepoDep,
    ver_repo: VersionRepoDep,
    chunking_service: ChunkingServiceDep,
    embedding_service: EmbeddingServiceDep,
    sec_ctx: SecurityContextDep,
    authz_policy: AuthzPolicyDep,
) -> DocumentProcessResponse:
    """Trigger chunking and embedding generation for a document with tenant authorization."""
    start_time = time.perf_counter()

    doc = await doc_repo.get_by_id(document_id)
    if not doc:
        raise EntityNotFoundError(entity_name="Document", entity_id=str(document_id))

    authz_policy.authorize_document(sec_ctx, doc.metadata_json, DocumentAction.PROCESS)

    latest_version = await ver_repo.get_latest_version(document_id)
    if not latest_version:
        raise EntityNotFoundError(
            entity_name="DocumentVersion",
            entity_id=f"for document {document_id}",
        )

    # 1. Chunk document
    chunks = await chunking_service.chunk_document_version(
        document_id=document_id,
        version_id=latest_version.id,
    )

    # 2. Embed document chunks
    vectors = await embedding_service.embed_document_version(
        document_id=document_id,
        version_id=latest_version.id,
    )

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    return DocumentProcessResponse(
        document_id=document_id,
        version_id=latest_version.id,
        status="EMBEDDED",
        total_chunks=len(chunks),
        total_vectors=len(vectors),
        message=f"Successfully processed document: {len(chunks)} chunks and {len(vectors)} vectors generated.",
        elapsed_ms=elapsed_ms,
    )


@router.get(
    "/{document_id}/jobs",
    response_model=list[IngestionJobResponse],
    status_code=status.HTTP_200_OK,
    summary="Get document ingestion jobs",
    description="Retrieve all processing jobs associated with a document.",
    dependencies=[Depends(check_rate_limit), Depends(verify_api_key)],
)
async def get_document_jobs(
    document_id: uuid.UUID,
    job_repo: JobRepoDep,
    doc_repo: DocumentRepoDep,
    sec_ctx: SecurityContextDep,
    authz_policy: AuthzPolicyDep,
) -> list[IngestionJobResponse]:
    """List execution jobs for a specific document with tenant authorization."""
    doc = await doc_repo.get_by_id(document_id)
    if not doc:
        raise EntityNotFoundError(entity_name="Document", entity_id=str(document_id))

    authz_policy.authorize_document(sec_ctx, doc.metadata_json, DocumentAction.READ)

    jobs = await job_repo.get_jobs_by_document(document_id)
    return [
        IngestionJobResponse(
            id=j.id,
            document_id=j.document_id,
            document_version_id=j.document_version_id,
            status=j.status,
            error_message=j.error_message,
            stages_completed=j.stages_completed,
            stats=j.stats,
            started_at=j.started_at,
            completed_at=j.completed_at,
            created_at=j.created_at,
        )
        for j in jobs
    ]


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete document",
    description="Cascade delete a document, its versions, metadata, and jobs.",
    dependencies=[Depends(check_rate_limit), Depends(verify_api_key)],
)
async def delete_document(
    document_id: uuid.UUID,
    doc_repo: DocumentRepoDep,
    storage: StorageDep,
    sec_ctx: SecurityContextDep,
    authz_policy: AuthzPolicyDep,
) -> None:
    """Delete document by ID with tenant authorization and physical asset cleanup (SEC-05)."""
    doc = await doc_repo.get_by_id(document_id)
    if not doc:
        raise EntityNotFoundError(entity_name="Document", entity_id=str(document_id))

    authz_policy.authorize_document(sec_ctx, doc.metadata_json, DocumentAction.DELETE)

    # SEC-05: Clean up physical file in storage
    try:
        await storage.delete(doc.source)
    except Exception as exc:
        logger.warning(f"Failed to delete physical asset at '{doc.source}': {exc}")

    deleted = await doc_repo.delete(document_id)
    if not deleted:
        raise EntityNotFoundError(entity_name="Document", entity_id=str(document_id))


@router.get(
    "/{document_id}/chunks",
    response_model=list[ChunkResponse],
    status_code=status.HTTP_200_OK,
    summary="Get document chunks",
    description="Retrieve all text chunks and token metrics for the latest version of a document.",
    dependencies=[Depends(check_rate_limit), Depends(verify_api_key)],
)
async def get_document_chunks(
    document_id: uuid.UUID,
    doc_repo: DocumentRepoDep,
    ver_repo: VersionRepoDep,
    chunk_repo: ChunkRepoDep,
    sec_ctx: SecurityContextDep,
    authz_policy: AuthzPolicyDep,
) -> list[ChunkResponse]:
    """Fetch all chunk records and content for a document."""
    doc = await doc_repo.get_by_id(document_id)
    if not doc:
        raise EntityNotFoundError(entity_name="Document", entity_id=str(document_id))

    authz_policy.authorize_document(sec_ctx, doc.metadata_json, DocumentAction.READ)

    latest_version = await ver_repo.get_latest_version(document_id)
    if not latest_version:
        return []

    chunks = await chunk_repo.get_chunks_by_version(latest_version.id)
    return [
        ChunkResponse(
            id=c.id,
            chunk_index=c.chunk_index,
            content=c.content,
            token_count=c.token_count,
            char_count=c.char_count,
            embedding_id=c.embedding_id,
            metadata_json=c.metadata_json,
        )
        for c in chunks
    ]


@router.get(
    "/{document_id}/content",
    summary="Open or download raw document content",
    description="Retrieve the raw stored file bytes for a document with inline viewing disposition.",
    dependencies=[Depends(check_rate_limit), Depends(verify_api_key)],
)
async def get_document_content(
    document_id: uuid.UUID,
    doc_repo: DocumentRepoDep,
    storage: StorageDep,
    sec_ctx: SecurityContextDep,
    authz_policy: AuthzPolicyDep,
) -> Response:
    """Stream raw document content bytes for inline viewing or download."""
    doc = await doc_repo.get_by_id(document_id)
    if not doc:
        raise EntityNotFoundError(entity_name="Document", entity_id=str(document_id))

    authz_policy.authorize_document(sec_ctx, doc.metadata_json, DocumentAction.READ)

    try:
        content_bytes = await storage.read(doc.source)
    except Exception as exc:
        raise EntityNotFoundError(
            entity_name="DocumentContent", entity_id=str(document_id)
        ) from exc

    media_type = "text/plain"
    ext = doc.document_type.lower()
    if ext == "pdf":
        media_type = "application/pdf"
    elif ext == "md":
        media_type = "text/markdown"
    elif ext == "docx":
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    headers = {
        "Content-Disposition": f'inline; filename="{doc.name}"',
    }
    return Response(content=content_bytes, media_type=media_type, headers=headers)
