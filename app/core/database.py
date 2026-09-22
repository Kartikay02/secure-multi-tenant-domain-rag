"""Database engine, connection pooling, and async session management."""

import contextvars
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger("app.database")

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
_current_session_ctx: contextvars.ContextVar[AsyncSession | None] = contextvars.ContextVar(
    "current_session_ctx", default=None
)


def get_current_session() -> AsyncSession | None:
    """Return the active task-scoped database session, if any."""
    return _current_session_ctx.get()


def set_current_session(session: AsyncSession | None) -> contextvars.Token[AsyncSession | None]:
    """Set the active task-scoped database session."""
    return _current_session_ctx.set(session)


def reset_current_session(token: contextvars.Token[AsyncSession | None]) -> None:
    """Reset the task-scoped database session context."""
    _current_session_ctx.reset(token)


def get_engine(settings: Settings | None = None) -> AsyncEngine:
    """Return the initialized async database engine singleton."""
    global _engine
    if _engine is None:
        cfg = settings or get_settings()
        is_sqlite = cfg.database.url.startswith("sqlite")

        engine_kwargs: dict[str, object] = {
            "echo": cfg.app.debug,
            "future": True,
        }

        # SQLite does not support standard connection pool size / max_overflow
        if not is_sqlite:
            engine_kwargs.update(
                {
                    "pool_size": cfg.database.pool_size,
                    "max_overflow": cfg.database.max_overflow,
                    "pool_timeout": cfg.database.pool_timeout,
                    "pool_pre_ping": True,
                }
            )

        _engine = create_async_engine(cfg.database.url, **engine_kwargs)
        logger.info(f"Initialized database engine for dialect: {_engine.dialect.name}")

    return _engine


def get_session_factory(settings: Settings | None = None) -> async_sessionmaker[AsyncSession]:
    """Return the initialized async session factory singleton."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine(settings)
        _session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            autoflush=False,
            expire_on_commit=False,
        )
    return _session_factory


@asynccontextmanager
async def get_db_session(
    settings: Settings | None = None,
) -> AsyncIterator[AsyncSession]:
    """Provide a transaction-safe async database session context.

    Automatically commits when the block exits cleanly, rolls back on exception,
    and closes the session.
    """
    factory = get_session_factory(settings)
    session: AsyncSession = factory()
    token = _current_session_ctx.set(session)
    try:
        yield session
        await session.commit()
    except Exception as exc:
        await session.rollback()
        logger.error(f"Database transaction rolled back due to error: {exc}")
        raise exc
    finally:
        _current_session_ctx.reset(token)
        await session.close()


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding an async database session."""
    async with get_db_session() as session:
        yield session


async def check_database_health(settings: Settings | None = None) -> bool:
    """Execute 'SELECT 1' to verify database connectivity."""
    try:
        async with get_db_session(settings) as session:
            result = await session.execute(text("SELECT 1"))
            return bool(result.scalar() == 1)
    except Exception as exc:
        logger.warning(f"Database health check failed: {exc}")
        return False


async def close_engine() -> None:
    """Dispose of engine connection pool during graceful shutdown."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database engine disposed.")
