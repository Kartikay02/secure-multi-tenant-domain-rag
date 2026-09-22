"""Global exception handlers mapping exceptions to standard RFC 7807 responses."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import AppException
from app.core.logging import get_correlation_id, get_logger
from app.schemas.common import ErrorDetail, ErrorResponse

logger = get_logger("app.error_handler")


def register_exception_handlers(app: FastAPI) -> None:
    """Register application-wide exception handlers with FastAPI."""

    @app.exception_handler(AppException)
    async def handle_app_exception(_: Request, exc: AppException) -> JSONResponse:
        correlation_id = get_correlation_id()
        logger.warning(
            f"Domain exception [{exc.error_code}]: {exc.message} (status={exc.status_code})"
        )

        error_response = ErrorResponse(
            title=exc.error_code.replace("_", " ").title(),
            status=exc.status_code,
            detail=exc.message,
            error_code=exc.error_code,
            correlation_id=correlation_id,
            errors=[],
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.model_dump(mode="json"),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_exception(_: Request, exc: RequestValidationError) -> JSONResponse:
        correlation_id = get_correlation_id()
        errors = [
            ErrorDetail(
                field=" -> ".join(str(loc) for loc in err.get("loc", [])),
                message=err.get("msg", "Invalid value"),
                type=err.get("type"),
            )
            for err in exc.errors()
        ]

        logger.warning(f"Request validation error on fields: {[e.field for e in errors]}")

        error_response = ErrorResponse(
            title="Request Validation Failed",
            status=422,
            detail="The request body or parameters failed validation schema requirements.",
            error_code="VALIDATION_ERROR",
            correlation_id=correlation_id,
            errors=errors,
        )
        return JSONResponse(
            status_code=422,
            content=error_response.model_dump(mode="json"),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        correlation_id = get_correlation_id()
        error_response = ErrorResponse(
            title="HTTP Error",
            status=exc.status_code,
            detail=str(exc.detail),
            error_code=f"HTTP_{exc.status_code}",
            correlation_id=correlation_id,
            errors=[],
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def handle_generic_exception(_: Request, exc: Exception) -> JSONResponse:
        correlation_id = get_correlation_id()
        logger.error(
            f"Unhandled server exception: {exc}",
            exc_info=True,
        )

        error_response = ErrorResponse(
            title="Internal Server Error",
            status=500,
            detail="An unexpected internal error occurred. Please contact system support.",
            error_code="INTERNAL_SERVER_ERROR",
            correlation_id=correlation_id,
            errors=[],
        )
        return JSONResponse(
            status_code=500,
            content=error_response.model_dump(mode="json"),
        )
