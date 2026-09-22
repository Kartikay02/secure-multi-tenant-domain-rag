"""Unit tests for section-based Pydantic Settings configuration."""

import pytest
from pydantic import ValidationError

from app.core.config import (
    ApplicationSettings,
    DatabaseSettings,
    EmbeddingSettings,
    LLMSettings,
    ObservabilitySettings,
    RetrievalSettings,
    SecuritySettings,
    Settings,
    VectorStoreSettings,
    get_settings,
)


def test_default_settings_instantiation() -> None:
    """Verify all 8 configuration sections initialize with valid defaults."""
    settings = Settings(database=DatabaseSettings(_env_file=None))  # type: ignore[call-arg]

    # 1. Application
    assert settings.app.name == "Domain RAG System"
    assert settings.app.env in ["development", "staging", "production", "test"]
    assert settings.app.port == 8000
    assert settings.app.api_v1_prefix == "/api/v1"

    # 2. Database
    assert "postgresql+asyncpg" in settings.database.url
    assert settings.database.port == 5432
    assert settings.database.pool_size == 10

    # 3. LLM
    assert settings.llm.provider in ["mock", "openai"]
    assert settings.llm.model == "gpt-4o-mini"
    assert settings.llm.temperature == 0.0

    # 4. Embeddings
    assert settings.embeddings.provider in ["mock", "fastembed", "openai"]
    assert settings.embeddings.dimension == 384
    assert settings.embeddings.batch_size == 32

    # 5. Vector Store
    assert settings.vector_store.provider in ["pgvector", "mock"]
    assert settings.vector_store.index_type == "hnsw"
    assert settings.vector_store.distance_metric == "cosine"

    # 6. Retrieval
    assert settings.retrieval.default_top_k == 5
    assert settings.retrieval.rrf_k == 60
    assert settings.retrieval.similarity_threshold == 0.65

    # 7. Security
    assert settings.security.api_key_header == "X-API-Key"
    assert settings.security.max_file_size_mb == 25

    # 8. Observability
    assert settings.observability.log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    assert settings.observability.log_format in ["json", "console"]
    assert settings.observability.enable_metrics is True


def test_section_custom_overrides() -> None:
    """Verify individual sections can be customized via keyword instantiation."""
    custom_settings = Settings(
        app=ApplicationSettings(name="Custom App", env="staging"),
        database=DatabaseSettings(port=5433),
        llm=LLMSettings(temperature=0.8),
        embeddings=EmbeddingSettings(dimension=1536),
        vector_store=VectorStoreSettings(hnsw_m=32),
        retrieval=RetrievalSettings(default_top_k=10),
        security=SecuritySettings(max_file_size_mb=50),
        observability=ObservabilitySettings(log_level="DEBUG"),
    )
    assert custom_settings.app.name == "Custom App"
    assert custom_settings.app.env == "staging"
    assert custom_settings.database.port == 5433
    assert custom_settings.llm.temperature == 0.8
    assert custom_settings.embeddings.dimension == 1536
    assert custom_settings.vector_store.hnsw_m == 32
    assert custom_settings.retrieval.default_top_k == 10
    assert custom_settings.security.max_file_size_mb == 50
    assert custom_settings.observability.log_level == "DEBUG"


def test_invalid_temperature_validation() -> None:
    """Verify LLM temperature must be within [0.0, 2.0]."""
    with pytest.raises(ValidationError):
        LLMSettings(temperature=2.5)

    with pytest.raises(ValidationError):
        LLMSettings(temperature=-0.1)


def test_invalid_environment_validation() -> None:
    """Verify application environment only accepts allowed literals."""
    with pytest.raises(ValidationError):
        ApplicationSettings(env="invalid_env")  # type: ignore[arg-type]


def test_invalid_similarity_threshold_validation() -> None:
    """Verify retrieval similarity threshold is bounded in [0.0, 1.0]."""
    with pytest.raises(ValidationError):
        RetrievalSettings(similarity_threshold=1.5)


def test_get_settings_caching() -> None:
    """Verify get_settings returns the same cached instance singleton."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_production_rejects_in_memory_rate_limiting() -> None:
    """Production mode must reject rate_limit_backend='memory' for multi-worker safety."""
    from app.core.config import validate_production_settings

    prod_settings = Settings(
        app=ApplicationSettings(env="production", debug=False),
        security=SecuritySettings(
            api_key="production-strong-secret-key-42chars-long",
            rate_limit_backend="memory",
        ),
        database=DatabaseSettings(url="postgresql+asyncpg://postgres:pass@localhost:5432/rag"),
        vector_store=VectorStoreSettings(provider="pgvector"),
        llm=LLMSettings(provider="openai", api_key="sk-valid-key-here-for-test-32chars"),
    )

    with pytest.raises(ValueError) as exc_info:
        validate_production_settings(prod_settings)

    assert "rate_limit_backend='memory' is not allowed in production" in str(exc_info.value)


def test_production_allows_redis_rate_limiting_with_password() -> None:
    """Production mode allows rate_limit_backend='redis' when password is configured."""
    from app.core.config import RedisSettings, validate_production_settings

    prod_settings = Settings(
        app=ApplicationSettings(env="production", debug=False),
        security=SecuritySettings(
            api_key="production-strong-secret-key-42chars-long",
            rate_limit_backend="redis",
        ),
        redis=RedisSettings(url="redis://:strongredispass@redis-host:6379/0"),
        database=DatabaseSettings(url="postgresql+asyncpg://postgres:pass@localhost:5432/rag"),
        vector_store=VectorStoreSettings(provider="pgvector"),
        llm=LLMSettings(provider="openai", api_key="sk-valid-key-here-for-test-32chars"),
    )

    # Should validate cleanly with no exceptions
    validate_production_settings(prod_settings)


def test_development_and_test_allow_in_memory_rate_limiting() -> None:
    """Development and test environments explicitly allow in-memory rate limiting."""
    from app.core.config import validate_production_settings

    dev_settings = Settings(
        app=ApplicationSettings(env="development"),
        security=SecuritySettings(rate_limit_backend="memory"),
    )
    validate_production_settings(dev_settings)

    test_settings = Settings(
        app=ApplicationSettings(env="test"),
        security=SecuritySettings(rate_limit_backend="memory"),
    )
    validate_production_settings(test_settings)
