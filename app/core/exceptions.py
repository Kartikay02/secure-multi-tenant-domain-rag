"""Domain and application exception hierarchy."""

from typing import Any


class AppException(Exception):
    """Base application exception with standardized error details."""

    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_SERVER_ERROR",
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}


class ConfigurationError(AppException):
    """Raised when application configuration or environment variables are invalid."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            status_code=500,
            details=details,
        )


class EntityNotFoundError(AppException):
    """Raised when a requested resource (document, chunk, query) does not exist."""

    def __init__(self, entity_name: str, entity_id: str) -> None:
        super().__init__(
            message=f"{entity_name} with ID '{entity_id}' not found.",
            error_code="NOT_FOUND",
            status_code=404,
            details={"entity_name": entity_name, "entity_id": entity_id},
        )


class DuplicateEntityError(AppException):
    """Raised when attempting to create an entity that violates uniqueness."""

    def __init__(self, entity_name: str, identifier: str) -> None:
        super().__init__(
            message=f"{entity_name} with identifier '{identifier}' already exists.",
            error_code="DUPLICATE_ENTITY",
            status_code=409,
            details={"entity_name": entity_name, "identifier": identifier},
        )


class IngestionError(AppException):
    """Raised when document ingestion fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            error_code="INGESTION_ERROR",
            status_code=422,
            details=details,
        )


class ParsingError(AppException):
    """Raised when document parsing or extraction fails."""

    def __init__(self, filename: str, reason: str) -> None:
        super().__init__(
            message=f"Failed to parse '{filename}': {reason}",
            error_code="PARSING_ERROR",
            status_code=422,
            details={"filename": filename, "reason": reason},
        )


class EmbeddingError(AppException):
    """Raised when generating embeddings fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            error_code="EMBEDDING_ERROR",
            status_code=502,
            details=details,
        )


class EmbeddingDimensionMismatchError(EmbeddingError):
    """Raised when returned embedding dimension does not match expected configuration."""

    def __init__(self, expected: int, received: int, model: str) -> None:
        super().__init__(
            message=(
                f"Embedding dimension mismatch for model '{model}': "
                f"expected {expected}, received {received}."
            ),
            details={"expected": expected, "received": received, "model": model},
        )
        self.error_code = "EMBEDDING_DIMENSION_MISMATCH"


class EmbeddingTimeoutError(EmbeddingError):
    """Raised when an embedding provider request times out."""

    def __init__(self, provider: str, timeout_seconds: float) -> None:
        super().__init__(
            message=f"Embedding request to provider '{provider}' timed out after {timeout_seconds}s.",
            details={"provider": provider, "timeout_seconds": timeout_seconds},
        )
        self.error_code = "EMBEDDING_TIMEOUT"
        self.status_code = 504


class EmbeddingRateLimitError(EmbeddingError):
    """Raised when embedding provider rate limits are exhausted after retries."""

    def __init__(self, provider: str, retry_after: float | None = None) -> None:
        super().__init__(
            message=f"Embedding provider '{provider}' rate limit exceeded. Retries exhausted.",
            details={"provider": provider, "retry_after": retry_after},
        )
        self.error_code = "EMBEDDING_RATE_LIMIT_EXCEEDED"
        self.status_code = 429


class EmbeddingAuthenticationError(EmbeddingError):
    """Raised when credentials for the embedding provider are invalid or missing."""

    def __init__(self, provider: str, reason: str = "Invalid API key or unauthorized.") -> None:
        super().__init__(
            message=f"Authentication failed for embedding provider '{provider}': {reason}",
            details={"provider": provider, "reason": reason},
        )
        self.error_code = "EMBEDDING_AUTHENTICATION_ERROR"
        self.status_code = 401


class RetrievalError(AppException):
    """Raised when retrieving candidate chunks fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            error_code="RETRIEVAL_ERROR",
            status_code=500,
            details=details,
        )


class RerankError(AppException):
    """Raised when cross-encoder reranking fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            error_code="RERANK_ERROR",
            status_code=502,
            details=details,
        )


class RerankTimeoutError(RerankError):
    """Raised when a reranking request exceeds allowed timeout."""

    def __init__(self, provider: str, timeout_seconds: float) -> None:
        super().__init__(
            message=f"Reranking request to provider '{provider}' timed out after {timeout_seconds}s.",
            details={"provider": provider, "timeout_seconds": timeout_seconds},
        )
        self.error_code = "RERANK_TIMEOUT"
        self.status_code = 504


class RerankAuthenticationError(RerankError):
    """Raised when authentication credentials for reranker are invalid."""

    def __init__(self, provider: str, reason: str = "Invalid API key or unauthorized.") -> None:
        super().__init__(
            message=f"Authentication failed for reranker provider '{provider}': {reason}",
            details={"provider": provider, "reason": reason},
        )
        self.error_code = "RERANK_AUTHENTICATION_ERROR"
        self.status_code = 401


class ContextError(AppException):
    """Raised when context assembly or document packing fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            error_code="CONTEXT_ERROR",
            status_code=500,
            details=details,
        )


class ContextBudgetExceededError(ContextError):
    """Raised when required minimum context exceeds configured token budget."""

    def __init__(self, required_tokens: int, max_tokens: int) -> None:
        super().__init__(
            message=f"Context tokens ({required_tokens}) exceeded maximum allowed budget ({max_tokens}).",
            details={"required_tokens": required_tokens, "max_tokens": max_tokens},
        )
        self.error_code = "CONTEXT_BUDGET_EXCEEDED"
        self.status_code = 400


class GenerationError(AppException):
    """Raised when LLM answer generation or streaming fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            error_code="GENERATION_ERROR",
            status_code=502,
            details=details,
        )


class GenerationTimeoutError(GenerationError):
    """Raised when an LLM provider request times out."""

    def __init__(self, provider: str, timeout_seconds: float) -> None:
        super().__init__(
            message=f"Generation request to provider '{provider}' timed out after {timeout_seconds}s.",
            details={"provider": provider, "timeout_seconds": timeout_seconds},
        )
        self.error_code = "GENERATION_TIMEOUT"
        self.status_code = 504


class GenerationRateLimitError(GenerationError):
    """Raised when LLM provider rate limits are exhausted after retries."""

    def __init__(self, provider: str, retry_after: float | None = None) -> None:
        super().__init__(
            message=f"LLM provider '{provider}' rate limit exceeded. Retries exhausted.",
            details={"provider": provider, "retry_after": retry_after},
        )
        self.error_code = "GENERATION_RATE_LIMIT_EXCEEDED"
        self.status_code = 429


class GenerationAuthenticationError(GenerationError):
    """Raised when credentials for the LLM provider are invalid or missing."""

    def __init__(self, provider: str, reason: str = "Invalid API key or unauthorized.") -> None:
        super().__init__(
            message=f"Authentication failed for LLM provider '{provider}': {reason}",
            details={"provider": provider, "reason": reason},
        )
        self.error_code = "GENERATION_AUTHENTICATION_ERROR"
        self.status_code = 401


class GuardrailViolationError(AppException):
    """Raised when generated response fails grounding or safety checks."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            error_code="GUARDRAIL_VIOLATION",
            status_code=422,
            details=details,
        )


class UnauthorizedError(AppException):
    """Raised when request lacks valid authentication."""

    def __init__(self, message: str = "Authentication required.") -> None:
        super().__init__(
            message=message,
            error_code="UNAUTHORIZED",
            status_code=401,
        )


class ForbiddenError(AppException):
    """Raised when caller lacks required permissions or tenant access."""

    def __init__(self, message: str = "Access denied.") -> None:
        super().__init__(
            message=message,
            error_code="FORBIDDEN",
            status_code=403,
        )


class SecurityValidationError(AppException):
    """Raised when request violates security constraints (e.g. prompt injection, malicious payload)."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            error_code="SECURITY_VALIDATION_ERROR",
            status_code=400,
            details=details,
        )


class RateLimitExceededError(AppException):
    """Raised when a client exceeds allowed request thresholds."""

    def __init__(self, retry_after_seconds: int = 60, message: str | None = None) -> None:
        self.retry_after_seconds = retry_after_seconds
        msg = message or f"Rate limit exceeded. Please retry after {retry_after_seconds} seconds."
        super().__init__(
            message=msg,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details={"retry_after_seconds": retry_after_seconds},
        )


class DatabaseError(AppException):
    """Raised when a persistence operation fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            status_code=500,
            details=details,
        )


class DatabaseConnectionError(AppException):
    """Raised when the database connection cannot be established or is lost."""

    def __init__(self, message: str = "Unable to connect to database backend.") -> None:
        super().__init__(
            message=message,
            error_code="DATABASE_CONNECTION_ERROR",
            status_code=503,
        )


class EntityConflictError(AppException):
    """Raised when a concurrent modification or version conflict occurs."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            error_code="ENTITY_CONFLICT",
            status_code=409,
            details=details,
        )


class FileValidationError(AppException):
    """Raised when an uploaded file violates safety or format constraints."""

    def __init__(
        self,
        message: str,
        error_code: str = "FILE_VALIDATION_ERROR",
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details,
        )


class UnsupportedFileTypeError(FileValidationError):
    """Raised when an uploaded file extension or MIME type is not permitted."""

    def __init__(self, extension: str, mime_type: str, allowed: list[str]) -> None:
        super().__init__(
            message=f"Unsupported file type '{extension}' ({mime_type}). Allowed types: {', '.join(allowed)}",
            error_code="UNSUPPORTED_FILE_TYPE",
            status_code=415,
            details={"extension": extension, "mime_type": mime_type, "allowed": allowed},
        )


class FileSizeLimitExceededError(FileValidationError):
    """Raised when an uploaded file exceeds the configured maximum size threshold."""

    def __init__(
        self,
        size_bytes: int,
        max_allowed_bytes: int | None = None,
        max_bytes: int | None = None,
    ) -> None:
        effective_max = max_allowed_bytes if max_allowed_bytes is not None else (max_bytes or 0)
        max_mb = round(effective_max / (1024 * 1024), 2)
        size_mb = round(size_bytes / (1024 * 1024), 2)
        super().__init__(
            message=f"File size ({size_mb} MB) exceeds maximum allowed limit of {max_mb} MB.",
            error_code="FILE_SIZE_LIMIT_EXCEEDED",
            status_code=413,
            details={"size_bytes": size_bytes, "max_allowed_bytes": effective_max},
        )


class StorageError(AppException):
    """Raised when reading, writing, or deleting files in storage fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            error_code="STORAGE_ERROR",
            status_code=500,
            details=details,
        )


class QueryValidationError(AppException):
    """Raised when a user query fails validation (empty, too short, or invalid)."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            error_code="QUERY_VALIDATION_ERROR",
            status_code=400,
            details=details,
        )


class PipelineStageError(AppException):
    """Raised when an unrecoverable failure occurs in a specific RAG pipeline stage."""

    def __init__(
        self,
        stage: str,
        message: str,
        request_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        err_details = details.copy() if details else {}
        err_details["stage"] = stage
        if request_id:
            err_details["request_id"] = request_id

        super().__init__(
            message=f"Pipeline stage '{stage}' failed: {message}",
            error_code="PIPELINE_STAGE_ERROR",
            status_code=500,
            details=err_details,
        )
        self.stage = stage
        self.request_id = request_id
