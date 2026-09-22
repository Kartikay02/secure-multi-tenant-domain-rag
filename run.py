"""Domain RAG Platform - Application Runner.

Usage:
    python run.py
"""

import asyncio
import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


async def init_db() -> None:
    import app.models.chunk  # noqa: F401
    import app.models.document  # noqa: F401
    import app.models.version  # noqa: F401
    from app.core.config import get_settings
    from app.core.database import get_engine
    from app.models.base import Base

    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    engine = get_engine(settings)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()


def print_banner(host: str, port: int, api_key: str) -> None:
    print(f"""
================================================================================
  DOMAIN RAG PLATFORM - LIVE SERVER
================================================================================
  Status:           RUNNING
  Local Web UI:     http://{host}:{port}/ui  (or http://{host}:{port}/)
  OpenAPI Docs:     http://{host}:{port}/api/v1/docs
  ReDoc Docs:       http://{host}:{port}/api/v1/redoc
  Liveness Check:   http://{host}:{port}/api/v1/health
  Readiness Check:  http://{host}:{port}/api/v1/ready
--------------------------------------------------------------------------------
  AUTHENTICATION API KEYS:
  Master (Admin):   {api_key}
  Tenant Alpha:     tenant-alpha-key-12345  (Tenant: tenant_alpha)
  Tenant Bravo:     tenant-bravo-key-67890  (Tenant: tenant_bravo)
================================================================================
""")


def main() -> None:
    asyncio.run(init_db())

    from app.core.config import get_settings
    settings = get_settings()

    host = settings.app.host
    port = settings.app.port
    api_key = settings.security.api_key

    print_banner(host, port, api_key)

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,
        log_level=settings.observability.log_level.lower(),
    )


if __name__ == "__main__":
    main()
