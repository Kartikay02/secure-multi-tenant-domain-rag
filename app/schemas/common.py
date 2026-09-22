"""Standardized schemas for API responses, errors, and health checks."""

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness check response schema."""

    status: str = Field(default="healthy", description="Application operational status")
    version: str = Field(..., description="Application semantic version")
    environment: str = Field(..., description="Active runtime environment")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Current server UTC timestamp",
    )


class ReadinessResponse(BaseModel):
    """Readiness check response schema with dependency statuses."""

    status: str = Field(default="ready", description="Overall readiness state")
    version: str = Field(..., description="Application semantic version")
    checks: dict[str, str] = Field(
        default_factory=dict,
        description="Map of component names to their health status (e.g. database: ok)",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Current server UTC timestamp",
    )


class ErrorDetail(BaseModel):
    """Individual field validation error detail."""

    field: str | None = Field(default=None, description="Request field that caused the error")
    message: str = Field(..., description="Detailed description of the validation failure")
    type: str | None = Field(default=None, description="Pydantic validation error code")


class ErrorResponse(BaseModel):
    """Standardized error envelope following RFC 7807 problem details."""

    title: str = Field(..., description="Human-readable summary of the error type")
    status: int = Field(..., description="HTTP status code")
    detail: str = Field(..., description="Specific explanation of the error occurrence")
    error_code: str = Field(..., description="Machine-readable domain error code")
    correlation_id: str = Field(..., description="Unique request tracing ID")
    errors: list[ErrorDetail] = Field(
        default_factory=list, description="List of granular validation errors if applicable"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of the error",
    )


class PaginatedResponse[T](BaseModel):
    """Generic pagination wrapper schema."""

    items: list[T] = Field(..., description="List of records for the current page")
    total: int = Field(..., description="Total count of records across all pages")
    page: int = Field(..., description="Current 1-based page number")
    page_size: int = Field(..., description="Number of records per page")
    total_pages: int = Field(..., description="Calculated total number of pages")
