from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_app_settings
from app.core.config import Settings
from app.core.database import get_db
from app.schemas.common import HealthResponse, ReadinessResponse

router = APIRouter()

SettingsDep = Annotated[Settings, Depends(get_app_settings)]


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    description="Returns OK if the service process is up and handling HTTP requests.",
)
async def check_liveness(
    settings: SettingsDep,
) -> HealthResponse:
    """Return basic health status of the application."""
    return HealthResponse(
        status="healthy",
        version=settings.app.version,
        environment=settings.app.env,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    status_code=status.HTTP_200_OK,
    summary="Readiness probe",
    description="Returns OK if the service and downstream dependencies are ready to accept traffic.",
)
async def check_readiness(
    response: Response,
    settings: SettingsDep,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> ReadinessResponse:
    """Return readiness status including live downstream database connectivity probe (SEC-46)."""
    checks: dict[str, str] = {
        "api": "ok",
        "configuration": "valid",
    }
    is_ready = True

    try:
        await session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        from app.core.config import scrub_credentials

        checks["database"] = f"unreachable: {scrub_credentials(str(exc))}"
        is_ready = False

    # SEC-57: Downstream Redis connectivity probe when Redis rate limiting is configured
    if settings.security.rate_limit_backend == "redis":
        try:
            import redis.asyncio as aioredis

            redis_client = aioredis.from_url(
                settings.redis.resolved_url,
                decode_responses=True,
                socket_timeout=settings.redis.timeout_seconds,
                socket_connect_timeout=settings.redis.timeout_seconds,
            )
            await redis_client.ping()
            await redis_client.aclose()
            checks["redis"] = "ok"
        except Exception as exc:
            from app.core.config import scrub_credentials

            checks["redis"] = f"unreachable: {scrub_credentials(str(exc))}"
            if settings.app.env in ("production", "staging"):
                is_ready = False

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(
            status="not_ready",
            version=settings.app.version,
            checks=checks,
        )

    return ReadinessResponse(
        status="ready",
        version=settings.app.version,
        checks=checks,
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    status_code=status.HTTP_200_OK,
    summary="Readiness probe (alias)",
    description="Alias for /ready for backwards compatibility.",
    include_in_schema=True,
)
async def check_readiness_alias(
    response: Response,
    settings: SettingsDep,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> ReadinessResponse:
    """Backwards-compatible readiness check alias."""
    return await check_readiness(response=response, settings=settings, session=session)
