"""Pydantic schemas for document ingestion contracts."""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class IngestionResponse(BaseModel):
    """Response returned upon document ingestion."""

    document_id: uuid.UUID = Field(..., description="Unique ID of the parent document")
    document_version_id: uuid.UUID = Field(..., description="Unique ID of the specific version")
    job_id: uuid.UUID = Field(..., description="Ingestion processing job ID")
    filename: str = Field(..., description="Sanitized filename")
    content_hash: str = Field(..., description="SHA-256 hash of raw file content")
    size_bytes: int = Field(..., description="File size in bytes")
    status: str = Field(..., description="Current processing status")
    is_duplicate: bool = Field(
        default=False, description="True if an identical file was already indexed"
    )
    extracted_title: str | None = Field(default=None, description="Title detected by parser")
    extracted_author: str | None = Field(default=None, description="Author detected by parser")
    word_count: int = Field(default=0, description="Total word count of normalized content")
    char_count: int = Field(default=0, description="Total character count of normalized content")
    estimated_tokens: int = Field(
        default=0, description="Estimated token count of normalized content"
    )
    storage_uri: str = Field(..., description="Persistent storage location URI")
    message: str = Field(..., description="Human-readable processing status summary")


class DocumentVersionResponse(BaseModel):
    """Schema representing a document version revision."""

    id: uuid.UUID
    version_number: int
    content_hash: str
    size_bytes: int
    status: str
    total_chunks: int
    metadata_json: dict[str, Any]
    created_at: datetime


class DocumentMetadataResponse(BaseModel):
    """Schema representing a typed document metadata entry."""

    key: str
    value: str
    value_type: str


class DocumentDetailResponse(BaseModel):
    """Detailed response for a Document entity."""

    id: uuid.UUID
    name: str
    document_type: str
    source: str
    description: str | None
    metadata_json: dict[str, Any]
    versions: list[DocumentVersionResponse]
    metadata_entries: list[DocumentMetadataResponse]
    created_at: datetime
    updated_at: datetime


class IngestionJobResponse(BaseModel):
    """Schema representing an ingestion job's execution state."""

    id: uuid.UUID
    document_id: uuid.UUID
    document_version_id: uuid.UUID | None
    status: str
    error_message: str | None
    stages_completed: dict[str, Any]
    stats: dict[str, Any]
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime


class DocumentListItemResponse(BaseModel):
    """Summarized document entity for paginated lists."""

    id: uuid.UUID
    name: str
    document_type: str
    source: str
    description: str | None = None
    created_at: datetime
    updated_at: datetime


class DocumentProcessResponse(BaseModel):
    """Response returned when document processing (chunking + embedding) completes."""

    document_id: uuid.UUID
    version_id: uuid.UUID
    status: str
    total_chunks: int
    total_vectors: int
    message: str
    elapsed_ms: float = 0.0


class ChunkResponse(BaseModel):
    """Schema representing a single document chunk."""

    id: uuid.UUID
    chunk_index: int
    content: str
    token_count: int
    char_count: int
    embedding_id: str | None = None
    metadata_json: dict[str, Any] = Field(default_factory=dict)
