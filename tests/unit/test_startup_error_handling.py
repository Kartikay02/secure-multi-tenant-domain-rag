"""Unit tests for lifespan startup error handling and fail-fast behavior (Item 3)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI

from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.main import lifespan


@pytest.mark.asyncio
async def test_lifespan_fails_fast_in_production_on_rag_init_error() -> None:
    """In production, RAG component failure during startup must raise and fail fast."""
    app = FastAPI()
    test_settings = Settings()
    test_settings.app.env = "production"
    test_settings.security.api_key = "production-super-strong-api-key-32-chars-long"
    test_settings.vector_store.provider = "pgvector"
    test_settings.database.url = "postgresql+asyncpg://postgres:postgres@localhost:5432/rag_db"
    test_settings.llm.provider = "openai"

    with patch("app.main.get_settings", return_value=test_settings):
        with patch("app.main.validate_production_settings"):
            with patch(
                "app.api.deps.init_rag_components",
                side_effect=ConfigurationError("Vector store index corrupt"),
            ):
                with pytest.raises(ConfigurationError):
                    async with lifespan(app):
                        pass


@pytest.mark.asyncio
async def test_lifespan_fails_fast_in_production_on_redis_error() -> None:
    """In production with Redis rate limiting, Redis connection failure must raise."""
    app = FastAPI()
    test_settings = Settings()
    test_settings.app.env = "production"
    test_settings.security.api_key = "production-super-strong-api-key-32-chars-long"
    test_settings.security.rate_limit_backend = "redis"
    test_settings.redis.password = "redis-strong-secret-12345"
    test_settings.vector_store.provider = "pgvector"
    test_settings.database.url = "postgresql+asyncpg://postgres:postgres@localhost:5432/rag_db"
    test_settings.llm.provider = "openai"

    mock_redis = MagicMock()
    mock_aioredis = MagicMock()
    mock_client = AsyncMock()
    mock_client.ping.side_effect = ConnectionError("Connection refused to redis:6379")
    mock_aioredis.from_url.return_value = mock_client
    mock_redis.asyncio = mock_aioredis

    with patch("app.main.get_settings", return_value=test_settings):
        with patch("app.main.validate_production_settings"):
            with patch("app.api.deps.init_rag_components"):
                with patch.dict(
                    "sys.modules", {"redis": mock_redis, "redis.asyncio": mock_aioredis}
                ):
                    with pytest.raises(ConfigurationError) as exc_info:
                        async with lifespan(app):
                            pass
                    assert "Redis rate limiter backend unreachable" in str(exc_info.value)


@pytest.mark.asyncio
async def test_lifespan_defers_rag_init_in_development() -> None:
    """In development, RAG component failure logs warning and permits deferred startup."""
    app = FastAPI()
    test_settings = Settings()
    test_settings.app.env = "development"

    with patch("app.main.get_settings", return_value=test_settings):
        with patch(
            "app.api.deps.init_rag_components",
            side_effect=RuntimeError("Database offline in dev"),
        ):
            # Should not raise
            async with lifespan(app):
                pass
