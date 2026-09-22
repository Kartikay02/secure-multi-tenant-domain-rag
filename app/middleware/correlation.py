"""Middleware to propagate and record unique X-Correlation-ID for each request."""

import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import set_correlation_id

CORRELATION_ID_HEADER = "X-Correlation-ID"


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Ensure every HTTP request has an X-Correlation-ID attached and logged."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Extract existing correlation ID or generate a new UUID4
        incoming_id = request.headers.get(CORRELATION_ID_HEADER)
        correlation_id = (
            incoming_id.strip() if incoming_id and incoming_id.strip() else uuid.uuid4().hex
        )

        # Store in context variable for structured loggers
        set_correlation_id(correlation_id)

        # Attach to request state for access in route handlers
        request.state.correlation_id = correlation_id

        # Process the request down the pipeline
        response = await call_next(request)

        # Echo correlation ID back in response headers
        response.headers[CORRELATION_ID_HEADER] = correlation_id
        return response
