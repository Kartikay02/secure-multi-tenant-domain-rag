"""Application configuration with modular section-based settings using Pydantic."""

import re
from functools import lru_cache
from typing import Literal
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def scrub_credentials(text: str) -> str:
    """Scrub sensitive credentials (passwords, tokens) from text, URLs, and exception messages."""
    if not text:
        return ""
    # Matches scheme://user:pass@host or scheme://:pass@host
    scrubbed = re.sub(r"://([^:@\s]+):([^@\s]+)@", r"://\1:***@", text)
    scrubbed = re.sub(r"://:([^@\s]+)@", r"://:***@", scrubbed)
    return scrubbed


def resolve_redis_url(
    url: str = "redis://localhost:6379/0",
    password: str | None = None,
    username: str | None = None,
) -> str:
    """Merge separate password and username into Redis URL safely without double encoding.

    If password is provided and url does not contain credentials, merges it cleanly.
    If url already contains credentials, preserves them unless explicit password is provided.
    """
    if not url:
        url = "redis://localhost:6379/0"

    parsed = urlsplit(url)
    existing_user = parsed.username
    existing_pwd = parsed.password
    host = parsed.hostname or "localhost"
    port = parsed.port

    # Determine final user and password
    final_user = username if (username is not None and username.strip() != "") else existing_user
    final_pwd = password if (password is not None and password.strip() != "") else existing_pwd

    if final_user or final_pwd is not None:
        if final_user and final_pwd is not None:
            user_quoted = quote(unquote(final_user), safe="")
            pwd_quoted = quote(unquote(final_pwd), safe="")
            auth_str = f"{user_quoted}:{pwd_quoted}"
        elif final_pwd is not None:
            pwd_quoted = quote(unquote(final_pwd), safe="")
            auth_str = f":{pwd_quoted}"
        elif final_user:
            user_quoted = quote(unquote(final_user), safe="")
            auth_str = f"{user_quoted}"
        else:
            auth_str = ""
    else:
        auth_str = ""

    host_str = host
    if port is not None:
        host_str = f"{host}:{port}"

    netloc = f"{auth_str}@{host_str}" if auth_str else host_str
    scheme = parsed.scheme or "redis"
    path = parsed.path or "/0"
    if not path.startswith("/"):
        path = f"/{path}"

    return urlunsplit((scheme, netloc, path, parsed.query, parsed.fragment))


class ApplicationSettings(BaseSettings):
    """Application identity and web runtime settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="APP_",
        extra="ignore",
    )

    name: str = Field(default="Domain RAG System", description="Human-readable application name")
    env: Literal["development", "staging", "production", "test"] = Field(
        default="development", description="Runtime environment"
    )
    debug: bool = Field(default=False, description="Enable FastAPI debug mode")
    host: str = Field(default="0.0.0.0", description="Bind host interface")
    port: int = Field(default=8000, description="Bind port")
    api_v1_prefix: str = Field(default="/api/v1", description="URL prefix for v1 API endpoints")
    version: str = Field(default="0.1.0", description="Semantic version string")


class DatabaseSettings(BaseSettings):
    """Relational and vector database persistence configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="DATABASE_",
        populate_by_name=True,
        extra="ignore",
    )

    url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/rag_db",
        validation_alias=AliasChoices("DATABASE_URL", "POSTGRES_URL"),
        description="Async SQLAlchemy database connection string",
    )
    host: str = Field(
        default="localhost",
        validation_alias=AliasChoices("DATABASE_HOST", "POSTGRES_HOST"),
        description="PostgreSQL server hostname",
    )
    port: int = Field(
        default=5432,
        validation_alias=AliasChoices("DATABASE_PORT", "POSTGRES_PORT"),
        description="PostgreSQL server port",
    )
    name: str = Field(
        default="rag_db",
        validation_alias=AliasChoices("DATABASE_NAME", "POSTGRES_DB"),
        description="Database name",
    )
    user: str = Field(
        default="postgres",
        validation_alias=AliasChoices("DATABASE_USER", "POSTGRES_USER"),
        description="Database username",
    )
    password: str = Field(
        default="postgres",
        validation_alias=AliasChoices("DATABASE_PASSWORD", "POSTGRES_PASSWORD"),
        description="Database password",
    )
    pool_size: int = Field(default=10, description="SQLAlchemy connection pool size")
    max_overflow: int = Field(default=20, description="SQLAlchemy connection pool max overflow")
    pool_timeout: int = Field(default=30, description="Pool acquisition timeout in seconds")


class LLMSettings(BaseSettings):
    """Language model provider and inference generation settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="LLM_",
        populate_by_name=True,
        extra="ignore",
    )

    provider: Literal["mock", "openai"] = Field(
        default="mock", description="LLM provider implementation"
    )
    model: str = Field(default="gpt-4o-mini", description="Model identifier")
    temperature: float = Field(
        default=0.0, ge=0.0, le=2.0, description="Generation temperature [0.0, 2.0]"
    )
    max_tokens: int = Field(default=1024, gt=0, description="Max generation output tokens")
    api_key: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY"),
        description="API key for LLM provider",
    )
    api_base: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for hosted LLM API",
    )
    timeout_seconds: float = Field(
        default=30.0, gt=0.0, description="Timeout in seconds for LLM generation requests"
    )
    max_retries: int = Field(
        default=2, ge=0, description="Maximum retry attempts for transient generation failures"
    )
    backoff_factor: float = Field(default=0.5, gt=0.0, description="Initial retry delay in seconds")
    max_backoff_seconds: float = Field(
        default=10.0, gt=0.0, description="Maximum ceiling for exponential retry delay"
    )


class EmbeddingSettings(BaseSettings):
    """Text embedding provider and dimensionality settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="EMBEDDING_",
        populate_by_name=True,
        extra="ignore",
    )

    provider: Literal["mock", "fastembed", "openai"] = Field(
        default="mock", description="Embedding provider implementation"
    )
    model: str = Field(
        default="BAAI/bge-small-en-v1.5", description="Embedding model identifier or path"
    )
    dimension: int = Field(default=384, gt=0, description="Output embedding vector dimension")
    batch_size: int = Field(default=32, gt=0, description="Batch size for vector encoding")
    api_key: str = Field(
        default="",
        validation_alias=AliasChoices("EMBEDDING_API_KEY", "OPENAI_API_KEY"),
        description="API key for external embedding provider",
    )
    api_base: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for hosted embedding API",
    )
    timeout_seconds: float = Field(
        default=30.0, gt=0.0, description="Timeout in seconds for embedding requests"
    )
    max_retries: int = Field(
        default=3, ge=0, description="Maximum retry attempts for transient embedding failures"
    )
    backoff_factor: float = Field(
        default=0.5, gt=0.0, description="Initial backoff delay in seconds for retries"
    )
    max_backoff_seconds: float = Field(
        default=10.0, gt=0.0, description="Maximum backoff ceiling in seconds"
    )


class VectorStoreSettings(BaseSettings):
    """Vector database indexing and distance metric settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="VECTOR_STORE_",
        populate_by_name=True,
        extra="ignore",
    )

    provider: Literal["pgvector", "mock"] = Field(
        default="pgvector", description="Vector storage backend"
    )
    index_type: Literal["hnsw", "ivfflat"] = Field(
        default="hnsw", description="Vector index algorithm"
    )
    distance_metric: Literal["cosine", "l2", "inner_product"] = Field(
        default="cosine", description="Distance metric for similarity search"
    )
    hnsw_m: int = Field(default=16, gt=0, description="HNSW graph max links per node (m)")
    hnsw_ef_construction: int = Field(
        default=64, gt=0, description="HNSW construction build exploration depth (ef_construction)"
    )


class RetrievalSettings(BaseSettings):
    """Retrieval, hybrid ranking, and reranker pipeline settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RETRIEVAL_",
        populate_by_name=True,
        extra="ignore",
    )

    default_top_k: int = Field(default=5, gt=0, description="Final number of retrieved chunks")
    dense_top_k: int = Field(default=30, gt=0, description="Candidates from dense vector search")
    sparse_top_k: int = Field(default=30, gt=0, description="Candidates from sparse lexical search")
    rrf_k: int = Field(default=60, gt=0, description="Reciprocal Rank Fusion smoothing constant")
    similarity_threshold: float = Field(
        default=0.65, ge=0.0, le=1.0, description="Minimum similarity cut-off score"
    )
    fusion_strategy: Literal["rrf", "linear", "relative_score"] = Field(
        default="rrf", description="Fusion algorithm for hybrid retrieval"
    )
    dense_weight: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Weight for dense retrieval in linear fusion"
    )
    sparse_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for sparse lexical retrieval in linear fusion",
    )
    reranker_enabled: bool = Field(default=True, description="Enable cross-encoder reranking stage")
    reranker_provider: Literal["mock", "cohere", "flashrank", "none"] = Field(
        default="mock", description="Reranker provider implementation"
    )
    reranker_model: str = Field(default="rerank-v3.5", description="Cross-encoder reranker model")
    reranker_candidates_k: int = Field(
        default=20, gt=0, description="Candidate chunks sent from retrieval to reranker"
    )
    reranker_top_k: int = Field(default=5, gt=0, description="Final context chunks after reranking")
    reranker_timeout_seconds: float = Field(
        default=10.0, gt=0.0, description="Timeout in seconds for reranker requests"
    )
    reranker_api_base: str = Field(
        default="https://api.cohere.com/v1",
        description="Base URL for hosted reranker API",
    )
    reranker_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("RETRIEVAL_RERANKER_API_KEY", "COHERE_API_KEY"),
        description="API key for cloud reranker",
    )
    max_context_tokens: int = Field(
        default=3000, gt=0, description="Maximum token budget for packed context"
    )


class ContextSettings(BaseSettings):
    """Context assembly, token budgeting, and citation formatting settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CONTEXT_",
        populate_by_name=True,
        extra="ignore",
    )

    max_tokens: int = Field(
        default=3000, gt=0, description="Maximum token budget for assembled prompt context"
    )
    max_chunks: int = Field(
        default=10, gt=0, description="Maximum number of chunks to include in context"
    )
    deduplication_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Jaccard similarity threshold for near-duplicate pruning",
    )
    ordering_strategy: Literal["relevance", "lost_in_the_middle", "document_order"] = Field(
        default="lost_in_the_middle", description="Intelligent chunk positioning strategy"
    )
    compression_strategy: Literal["none", "whitespace", "extractive"] = Field(
        default="none", description="Context compression algorithm"
    )
    format_style: Literal["markdown", "xml"] = Field(
        default="markdown", description="Delimited block formatting style"
    )
    encoding_name: str = Field(
        default="cl100k_base", description="Tiktoken encoding model for token bounding"
    )


class GuardrailSettings(BaseSettings):
    """Grounding validation, hallucination detection, and citation guardrails."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="GUARDRAIL_",
        populate_by_name=True,
        extra="ignore",
    )

    enabled: bool = Field(default=True, description="Enable grounding validation guardrails")
    mandatory: bool = Field(
        default=True,
        description="Enforce mandatory server-side grounding validation; client enable_grounding=false cannot bypass this",
    )
    grounding_threshold: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Minimum ratio of supported claims required"
    )
    confidence_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Minimum overall confidence score for accepted answers",
    )
    min_claim_overlap: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Minimum lexical/semantic overlap required per claim",
    )
    require_citations: bool = Field(
        default=True,
        description="Enforce that factual claims must contain valid citation identifiers",
    )
    fallback_mode: Literal["refusal", "annotate", "raise"] = Field(
        default="refusal", description="Fallback behavior when grounding validation fails"
    )
    fallback_refusal_message: str = Field(
        default="I cannot answer this question with sufficient confidence based on the available documentation.",
        description="Standardized fallback answer when validation fails",
    )


class SecuritySettings(BaseSettings):
    """Application security, authorization, and payload limit settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SECURITY_",
        populate_by_name=True,
        extra="ignore",
    )

    api_key_header: str = Field(
        default="X-API-Key", description="HTTP header name for API key authentication"
    )
    api_key: str = Field(
        default="",
        validation_alias=AliasChoices("SECURITY_API_KEY", "API_KEY"),
        description="Master API key for protected routes",
    )
    rotation_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("SECURITY_ROTATION_API_KEY"),
        description="Secondary active master API key permitted during key rotation window",
    )
    tenant_api_keys: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of API keys to authorized tenant IDs (key -> tenant_id)",
    )
    revoked_api_keys: list[str] = Field(
        default_factory=list,
        description="List of revoked API keys or fingerprints that must be immediately denied",
    )
    trusted_proxies: list[str] = Field(
        default_factory=lambda: ["127.0.0.1", "::1"],
        description="IP addresses of trusted reverse proxies permitted to forward X-Forwarded-For",
    )
    rate_limit_backend: Literal["memory", "redis"] = Field(
        default="memory",
        description="Rate limiting backend store (memory for dev/test, redis for multi-worker prod)",
    )
    allowed_origins: list[str] = Field(default=["*"], description="CORS allowed origin list")
    allow_credentials: bool = Field(
        default=False, description="Whether to allow credentials in CORS headers"
    )
    allowed_methods: list[str] = Field(
        default_factory=lambda: ["GET", "POST", "DELETE", "OPTIONS"],
        description="Allowed HTTP methods for CORS preflight",
    )
    allowed_headers: list[str] = Field(
        default_factory=lambda: [
            "Content-Type",
            "Authorization",
            "X-API-Key",
            "X-Tenant-ID",
            "X-Correlation-ID",
        ],
        description="Allowed request headers for CORS preflight",
    )
    max_file_size_mb: int = Field(
        default=25, gt=0, description="Maximum allowed file upload size in MB"
    )
    prompt_injection_detection_enabled: bool = Field(
        default=True, description="Enable prompt injection detection heuristics on queries"
    )
    prompt_injection_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for blocking/flagging prompt injection",
    )
    prompt_injection_action: Literal["block", "sanitize", "log"] = Field(
        default="block", description="Defensive action when prompt injection is detected"
    )
    tenant_enforcement_enabled: bool = Field(
        default=True,
        description="Enforce tenant-level authorization boundaries on documents and queries",
    )
    max_decompression_ratio: float = Field(
        default=20.0,
        gt=1.0,
        description="Maximum allowed archive decompression ratio to prevent zip bombs",
    )
    max_uncompressed_size_mb: int = Field(
        default=50, gt=0, description="Maximum uncompressed size in MB for archives (.docx)"
    )
    max_zip_entries: int = Field(
        default=1000, gt=0, description="Maximum number of files allowed within an archive"
    )
    enable_security_headers: bool = Field(
        default=True, description="Inject OWASP recommended security headers on HTTP responses"
    )


class RedisSettings(BaseSettings):
    """Redis cache and distributed rate limiting configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="REDIS_",
        populate_by_name=True,
        extra="ignore",
    )

    url: str = Field(
        default="redis://localhost:6379/0",
        validation_alias=AliasChoices("REDIS_URL"),
        description="Redis connection URL",
    )
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    password: str = Field(
        default="",
        validation_alias=AliasChoices("REDIS_PASSWORD"),
        description="Redis password for authentication",
    )
    db: int = Field(default=0, description="Redis database index")
    timeout_seconds: float = Field(default=5.0, gt=0.0, description="Redis socket timeout")

    @property
    def resolved_url(self) -> str:
        """Resolved Redis URL merging separate password/host/port without double encoding."""
        return resolve_redis_url(self.url, password=self.password)

    @property
    def sanitized_url(self) -> str:
        """Sanitized Redis URL with password masked for safe logging and metrics."""
        return scrub_credentials(self.resolved_url)


class ObservabilitySettings(BaseSettings):
    """Logging, metrics, and tracing telemetry settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="OBSERVABILITY_",
        populate_by_name=True,
        extra="ignore",
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        validation_alias=AliasChoices("OBSERVABILITY_LOG_LEVEL", "LOG_LEVEL"),
        description="Logging level threshold",
    )
    log_format: Literal["json", "console"] = Field(
        default="console",
        validation_alias=AliasChoices("OBSERVABILITY_LOG_FORMAT", "LOG_FORMAT"),
        description="Log emission format (json for production, console for dev)",
    )
    enable_metrics: bool = Field(
        default=True, description="Enable latency and throughput metrics tracking"
    )
    enable_tracing: bool = Field(
        default=False, description="Enable OpenTelemetry distributed tracing"
    )


class OrchestrationSettings(BaseSettings):
    """Orchestrator pipeline stage execution settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="ORCHESTRATION_",
        populate_by_name=True,
        extra="ignore",
    )

    min_query_length: int = Field(default=2, ge=1, description="Minimum query character length")
    max_query_length: int = Field(default=2000, gt=0, description="Maximum query character length")
    enable_preprocessing: bool = Field(default=True, description="Enable query preprocessing stage")
    enable_reranking: bool = Field(default=True, description="Enable cross-encoder reranking stage")
    enable_grounding: bool = Field(default=True, description="Enable grounding validation stage")
    retrieval_top_k: int = Field(
        default=10, gt=0, description="Default retrieval top-k candidate chunks"
    )
    rerank_top_k: int = Field(default=5, gt=0, description="Default reranked final chunks")


class Settings(BaseSettings):
    """Root configuration aggregating all domain subsystem settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_nested_delimiter="__",
        extra="ignore",
    )

    app: ApplicationSettings = Field(default_factory=ApplicationSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    guardrails: GuardrailSettings = Field(default_factory=GuardrailSettings)
    orchestration: OrchestrationSettings = Field(default_factory=OrchestrationSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    @property
    def vector_store_provider(self) -> str:
        """Convenience property for vector store provider selection (SEC-43)."""
        return self.vector_store.provider


def validate_production_settings(settings: Settings) -> None:
    """Validate that production and staging configurations conform to security and compatibility constraints."""
    is_prod = settings.app.env == "production"
    is_prod_or_staging = settings.app.env in ("production", "staging")

    if is_prod_or_staging:
        # SEC-11 & SEC-32: Production/staging authentication must fail closed
        api_key = settings.security.api_key.strip()
        insecure_placeholders = {
            "production-secret-api-key-change-me",
            "test-api-key",
            "change-me",
            "dev-secret-key-change-in-production-32bytes-min",
            "secret",
            "admin",
            "password",
        }
        if not api_key:
            raise ValueError(
                f"Production configuration error: SECURITY_API_KEY must be set and cannot be empty when APP_ENV={settings.app.env}."
            )
        if api_key in insecure_placeholders or len(api_key) < 16:
            raise ValueError(
                f"Production configuration error: SECURITY_API_KEY contains an insecure default placeholder or is too short (<16 chars) in {settings.app.env}."
            )

        # SEC-12: Debug mode must be disabled in production
        if is_prod and settings.app.debug:
            raise ValueError(
                "Production configuration error: APP_DEBUG must be False when APP_ENV=production."
            )

        # SEC-08 & SEC-18: CORS production validation
        if "*" in settings.security.allowed_origins and settings.security.allow_credentials:
            raise ValueError(
                f"Production configuration error: CORS allowed_origins cannot be ['*'] when allow_credentials is True in {settings.app.env}."
            )

        # SEC-13 & SEC-54: Provider / Database compatibility
        if settings.vector_store.provider == "pgvector" and not settings.database.url.startswith(
            ("postgresql", "postgres")
        ):
            raise ValueError(
                f"Configuration mismatch: VECTOR_STORE_PROVIDER='pgvector' requires a PostgreSQL database URL, got '{settings.database.url}'."
            )

        if settings.vector_store.provider == "mock" and is_prod:
            raise ValueError(
                "Production configuration error: VECTOR_STORE_PROVIDER cannot be 'mock' when APP_ENV=production."
            )

        if settings.llm.provider == "mock":
            raise ValueError(
                f"Production configuration error: LLM_PROVIDER cannot be 'mock' when APP_ENV={settings.app.env}."
            )

        # SEC-57: Redis production security & availability validation (applies when Redis rate limiting is required)
        if settings.security.rate_limit_backend == "redis":
            resolved_url = settings.redis.resolved_url
            parsed_redis = urlsplit(resolved_url)
            if not parsed_redis.password:
                raise ValueError(
                    f"Production security error: Redis authentication (password) is required when Redis rate-limiting is active in {settings.app.env}."
                )

        if is_prod and settings.security.rate_limit_backend == "memory":
            raise ValueError(
                "Production configuration error: rate_limit_backend='memory' is not allowed in production; "
                "distributed rate limiting requires Redis ('SECURITY_RATE_LIMIT_BACKEND=redis') for multi-worker safety."
            )

        if str(settings.security.rate_limit_backend).lower() == "noop" and is_prod:
            raise ValueError(
                "Production configuration error: Rate limiting cannot be 'noop' in production."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
