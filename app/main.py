"""FastAPI application factory and lifecycle definition."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response

from app.api.v1.endpoints import health, metrics
from app.api.v1.router import api_router
from app.core.config import Settings, get_settings, validate_production_settings
from app.core.logging import get_logger, setup_logging
from app.middleware.error_handler import register_exception_handlers
from app.observability.middleware import ObservabilityMiddleware
from app.security.headers import SecurityHeadersMiddleware

logger = get_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown events."""
    settings = get_settings()
    setup_logging(
        log_level=settings.observability.log_level,
        log_format=settings.observability.log_format,
    )

    # SEC-11 / SEC-12 / SEC-13 / SEC-18 / SEC-41 / SEC-54: Validate production security invariants
    validate_production_settings(settings)

    logger.info(
        f"Starting {settings.app.name} [v{settings.app.version}] in '{settings.app.env}' environment"
    )

    is_prod_or_staging = settings.app.env in ("production", "staging")

    # Issue 2 & SEC-57: Initialize singleton RAG orchestrator and expensive components
    try:
        from app.api.deps import init_rag_components

        init_rag_components(settings)
    except Exception as exc:
        if is_prod_or_staging:
            logger.error(
                f"Critical startup failure during RAG initialization in '{settings.app.env}': {exc}",
                exc_info=True,
            )
            raise
        logger.warning(f"Deferred RAG orchestrator initialization: {exc}")

    # SEC-57: In production/staging, verify Redis connectivity if Redis is configured as rate-limiting backend
    if is_prod_or_staging and settings.security.rate_limit_backend == "redis":
        try:
            import redis.asyncio as aioredis

            test_client = aioredis.from_url(
                settings.redis.resolved_url,
                decode_responses=True,
                socket_timeout=settings.redis.timeout_seconds,
                socket_connect_timeout=settings.redis.timeout_seconds,
            )
            await test_client.ping()
            await test_client.aclose()
        except Exception as exc:
            from app.core.config import scrub_credentials
            from app.core.exceptions import ConfigurationError

            clean_err = scrub_credentials(str(exc))
            logger.error(
                f"Critical startup failure: Redis rate limiter backend unreachable in '{settings.app.env}': {clean_err}"
            )
            raise ConfigurationError(
                f"Redis rate limiter backend unreachable in '{settings.app.env}': {clean_err}"
            ) from None

    yield

    logger.info(f"Shutting down {settings.app.name}")
    try:
        from app.api.deps import shutdown_rag_components

        shutdown_rag_components()
    except Exception as exc:
        logger.warning(f"Error during RAG orchestrator shutdown: {exc}")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app_settings = settings or get_settings()

    app = FastAPI(
        title=app_settings.app.name,
        version=app_settings.app.version,
        description="Production-grade Domain RAG System API",
        openapi_url=f"{app_settings.app.api_v1_prefix}/openapi.json",
        docs_url=f"{app_settings.app.api_v1_prefix}/docs",
        redoc_url=f"{app_settings.app.api_v1_prefix}/redoc",
        lifespan=lifespan,
    )

    # 1. ObservabilityMiddleware (correlation ID, trace context, request timing, metrics, secret scrubbing)
    # 2. CORSMiddleware (SEC-08 / SEC-18: Strict CORS configuration)
    # 3. SecurityHeadersMiddleware (OWASP security response headers)
    app.add_middleware(
        SecurityHeadersMiddleware,
        enabled=app_settings.security.enable_security_headers,
        trusted_proxies=app_settings.security.trusted_proxies,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.security.allowed_origins,
        allow_credentials=app_settings.security.allow_credentials,
        allow_methods=app_settings.security.allowed_methods,
        allow_headers=app_settings.security.allowed_headers,
    )
    app.add_middleware(ObservabilityMiddleware)

    # Register standardized RFC 7807 error handlers
    register_exception_handlers(app)

    # Include root health checks (Kubernetes liveness/readiness probes)
    app.include_router(health.router, tags=["System Health"])
    # Include root metrics endpoint (Kubernetes Prometheus scraping)
    app.include_router(metrics.router, prefix="/metrics", tags=["Metrics & Observability"])

    # Include Version 1 API routes
    app.include_router(api_router, prefix=app_settings.app.api_v1_prefix)

    # Path to interactive Web UI
    ui_html_path = Path(__file__).parent / "ui" / "index.html"

    # Route root path and /ui directly to interactive Web UI
    @app.get("/", include_in_schema=False)
    @app.get("/ui", include_in_schema=False)
    async def serve_ui() -> Response:
        if ui_html_path.exists():
            return FileResponse(ui_html_path, media_type="text/html")
        return RedirectResponse(url=f"{app_settings.app.api_v1_prefix}/docs")

    return app


app = create_app()
