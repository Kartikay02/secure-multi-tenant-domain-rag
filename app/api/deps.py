"""FastAPI dependency injection providers for repositories, services, auth, and governance."""

from typing import Annotated, Any

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import APIKeyAuthenticator, AuthenticationProviderProtocol
from app.api.rate_limiting import (
    InMemoryRateLimiter,
    RateLimiterProtocol,
    RedisRateLimiter,
    get_client_identifier,
)
from app.core.config import Settings, get_settings
from app.core.database import get_db, set_current_session
from app.observability.hooks import get_telemetry_hook
from app.observability.interfaces import MetricsCollectorProtocol, TelemetryHookProtocol
from app.observability.metrics import get_metrics_collector
from app.rag.embeddings.factory import EmbeddingProviderFactory
from app.rag.generation.factory import LLMFactory
from app.rag.reranking.factory import RerankerFactory
from app.rag.retrieval.dense import DenseRetriever
from app.rag.retrieval.hybrid import HybridRetriever
from app.rag.retrieval.lexical import PGLexicalRetriever
from app.rag.storage.interfaces import FileStorageProtocol
from app.rag.storage.local import LocalStorageService
from app.rag.vector.factory import VectorStoreFactory
from app.rag.vector.interfaces import VectorStoreProtocol
from app.repositories import (
    SQLAlchemyDocumentChunkRepository,
    SQLAlchemyDocumentRepository,
    SQLAlchemyDocumentVersionRepository,
    SQLAlchemyIngestionJobRepository,
)
from app.repositories.interfaces.chunk import DocumentChunkRepositoryProtocol
from app.repositories.interfaces.document import DocumentRepositoryProtocol
from app.repositories.interfaces.job import IngestionJobRepositoryProtocol
from app.repositories.interfaces.version import DocumentVersionRepositoryProtocol
from app.security.authorization import (
    AuthorizationPolicyProtocol,
    SecurityContext,
    TenantDocumentAccessControl,
)
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.ingestion_service import IngestionService
from app.services.rag_service import RAGOrchestratorService, create_rag_orchestrator

_storage_service: FileStorageProtocol | None = None
_rate_limiter: RateLimiterProtocol | None = None
_authenticator: AuthenticationProviderProtocol | None = None


def get_app_settings() -> Settings:
    """Dependency provider for application configuration."""
    return get_settings()


def get_storage_service() -> FileStorageProtocol:
    """Dependency provider for file storage backend."""
    global _storage_service
    if _storage_service is None:
        _storage_service = LocalStorageService(base_dir="data/uploads")
    return _storage_service


def get_document_repository(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentRepositoryProtocol:
    """Dependency provider for Document repository."""
    return SQLAlchemyDocumentRepository(session=session)


def get_version_repository(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentVersionRepositoryProtocol:
    """Dependency provider for DocumentVersion repository."""
    return SQLAlchemyDocumentVersionRepository(session=session)


def get_chunk_repository(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentChunkRepositoryProtocol:
    """Dependency provider for DocumentChunk repository."""
    return SQLAlchemyDocumentChunkRepository(session=session)


def get_job_repository(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> IngestionJobRepositoryProtocol:
    """Dependency provider for IngestionJob repository."""
    return SQLAlchemyIngestionJobRepository(session=session)


def get_ingestion_service(
    doc_repo: Annotated[DocumentRepositoryProtocol, Depends(get_document_repository)],
    ver_repo: Annotated[DocumentVersionRepositoryProtocol, Depends(get_version_repository)],
    job_repo: Annotated[IngestionJobRepositoryProtocol, Depends(get_job_repository)],
    storage: Annotated[FileStorageProtocol, Depends(get_storage_service)],
) -> IngestionService:
    """Dependency provider for document IngestionService."""
    return IngestionService(
        document_repo=doc_repo,
        version_repo=ver_repo,
        job_repo=job_repo,
        storage_service=storage,
    )


def get_chunking_service(
    doc_repo: Annotated[DocumentRepositoryProtocol, Depends(get_document_repository)],
    ver_repo: Annotated[DocumentVersionRepositoryProtocol, Depends(get_version_repository)],
    chunk_repo: Annotated[DocumentChunkRepositoryProtocol, Depends(get_chunk_repository)],
    job_repo: Annotated[IngestionJobRepositoryProtocol, Depends(get_job_repository)],
) -> ChunkingService:
    """Dependency provider for document ChunkingService."""
    return ChunkingService(
        document_repo=doc_repo,
        version_repo=ver_repo,
        chunk_repo=chunk_repo,
        job_repo=job_repo,
    )


_embedding_provider: Any = None


def get_embedding_provider(
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> Any:
    """Dependency provider returning singleton, concurrency-safe EmbeddingProvider instance."""
    global _embedding_provider
    if _embedding_provider is None:
        _embedding_provider = EmbeddingProviderFactory.create(settings.embeddings)
    return _embedding_provider


def get_embedding_service(
    doc_repo: Annotated[DocumentRepositoryProtocol, Depends(get_document_repository)],
    ver_repo: Annotated[DocumentVersionRepositoryProtocol, Depends(get_version_repository)],
    chunk_repo: Annotated[DocumentChunkRepositoryProtocol, Depends(get_chunk_repository)],
    job_repo: Annotated[IngestionJobRepositoryProtocol, Depends(get_job_repository)],
    session: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_app_settings)],
    embedding_provider: Annotated[Any, Depends(get_embedding_provider)],
) -> EmbeddingService:
    """Dependency provider for EmbeddingService reusing the singleton EmbeddingProvider."""
    vector_store = VectorStoreFactory.create(settings=settings, session=session)
    return EmbeddingService(
        document_repo=doc_repo,
        version_repo=ver_repo,
        chunk_repo=chunk_repo,
        job_repo=job_repo,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )


_rag_orchestrator: RAGOrchestratorService | None = None
_vector_store: VectorStoreProtocol | None = None
_dense_retriever: DenseRetriever | None = None
_lexical_retriever: PGLexicalRetriever | None = None
_hybrid_retriever: HybridRetriever | None = None
_reranker: Any = None
_llm_provider: Any = None


def init_rag_components(settings: Settings | None = None) -> RAGOrchestratorService:
    """Initialize singleton RAG orchestrator and expensive components (Issue 2).

    Initialized once during application lifespan. Database sessions are dynamically
    resolved per-request via contextvars rather than sharing an AsyncSession instance
    across concurrent requests.
    """
    global _rag_orchestrator, _embedding_provider, _vector_store
    global _dense_retriever, _lexical_retriever, _hybrid_retriever
    global _reranker, _llm_provider

    cfg = settings or get_settings()

    _embedding_provider = EmbeddingProviderFactory.create(cfg.embeddings)
    _vector_store = VectorStoreFactory.create(settings=cfg, session=None)
    _dense_retriever = DenseRetriever(
        embedding_provider=_embedding_provider,
        vector_store=_vector_store,
    )
    _lexical_retriever = PGLexicalRetriever(session=None)
    _hybrid_retriever = HybridRetriever(
        dense_retriever=_dense_retriever,
        lexical_retriever=_lexical_retriever,
        dense_top_k=cfg.retrieval.dense_top_k,
        sparse_top_k=cfg.retrieval.sparse_top_k,
        final_top_k=cfg.retrieval.default_top_k,
    )
    _reranker = RerankerFactory.create_reranker(cfg.retrieval)
    _llm_provider = LLMFactory.create_provider(cfg.llm)

    _rag_orchestrator = create_rag_orchestrator(
        retriever=_hybrid_retriever,
        reranker=_reranker,
        llm_provider=_llm_provider,
        settings=cfg,
    )
    return _rag_orchestrator


def shutdown_rag_components() -> None:
    """Tear down and reset singleton RAG components."""
    global _rag_orchestrator, _embedding_provider, _vector_store
    global _dense_retriever, _lexical_retriever, _hybrid_retriever
    global _reranker, _llm_provider

    _rag_orchestrator = None
    _embedding_provider = None
    _vector_store = None
    _dense_retriever = None
    _lexical_retriever = None
    _hybrid_retriever = None
    _reranker = None
    _llm_provider = None


def get_rag_orchestrator(
    settings: Annotated[Settings, Depends(get_app_settings)],
    session: Annotated[AsyncSession | None, Depends(get_db)] = None,
) -> RAGOrchestratorService:
    """Dependency provider returning the singleton RAGOrchestratorService instance."""
    global _rag_orchestrator
    if session is not None:
        set_current_session(session)
    if _rag_orchestrator is None:
        _rag_orchestrator = init_rag_components(settings)
    return _rag_orchestrator


def get_api_key_authenticator(
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> AuthenticationProviderProtocol:
    """Dependency provider for API key authentication service."""
    global _authenticator
    if _authenticator is None:
        _authenticator = APIKeyAuthenticator(
            api_key=settings.security.api_key,
            header_name=settings.security.api_key_header,
            tenant_api_keys=settings.security.tenant_api_keys,
            revoked_api_keys=settings.security.revoked_api_keys,
            rotation_api_key=settings.security.rotation_api_key,
            app_env=settings.app.env,
        )
    return _authenticator


def verify_api_key(
    request: Request,
    auth: Annotated[AuthenticationProviderProtocol, Depends(get_api_key_authenticator)],
) -> str | None:
    """Dependency enforcing API key authentication when configured."""
    return auth.authenticate(request)


def get_rate_limiter(
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> RateLimiterProtocol:
    """Dependency provider for rate limiting service."""
    global _rate_limiter
    if _rate_limiter is None:
        if settings.security.rate_limit_backend == "redis":
            _rate_limiter = RedisRateLimiter(
                redis_url=settings.redis.url,
                password=settings.redis.password,
                max_requests=60,
                window_seconds=60,
                fail_closed=(settings.app.env in ("production", "staging")),
            )
        else:
            _rate_limiter = InMemoryRateLimiter(max_requests=60, window_seconds=60)
    return _rate_limiter


async def check_rate_limit(
    request: Request,
    limiter: Annotated[RateLimiterProtocol, Depends(get_rate_limiter)],
    settings: Annotated[Settings, Depends(get_app_settings)],
    auth: Annotated[AuthenticationProviderProtocol, Depends(get_api_key_authenticator)],
) -> None:
    """Dependency enforcing sliding-window request rate limiting.

    If request carries valid authentication credentials, rate limits by validated
    identity/tenant. If credentials are unauthenticated, invalid, or missing,
    rate limits strictly by trusted client IP so attackers cannot rotate arbitrary
    headers to bypass rate limiting.
    """
    authenticated_identity: str | None = None
    try:
        if hasattr(auth, "resolve_security_context"):
            sec_ctx = auth.resolve_security_context(request)
            if sec_ctx.is_authenticated:
                authenticated_identity = f"tenant:{sec_ctx.tenant_id}:user:{sec_ctx.user_id}"
        else:
            caller_id = auth.authenticate(request)
            if caller_id and caller_id != "anonymous":
                authenticated_identity = f"user:{caller_id}"
    except Exception:
        authenticated_identity = None

    key = get_client_identifier(
        request,
        trusted_proxies=settings.security.trusted_proxies,
        authenticated_identity=authenticated_identity,
    )
    await limiter.check_rate_limit(key)


def get_telemetry_hook_dep() -> TelemetryHookProtocol:
    """Dependency provider for telemetry hook."""
    return get_telemetry_hook()


def get_metrics_collector_dep() -> MetricsCollectorProtocol:
    """Dependency provider for metrics collector."""
    return get_metrics_collector()


_authz_policy: AuthorizationPolicyProtocol | None = None


def get_authorization_policy(
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> AuthorizationPolicyProtocol:
    """Dependency provider for document authorization policy."""
    global _authz_policy
    if _authz_policy is None:
        _authz_policy = TenantDocumentAccessControl(
            tenant_enforcement_enabled=settings.security.tenant_enforcement_enabled
        )
    return _authz_policy


def get_security_context(
    request: Request,
    auth: Annotated[AuthenticationProviderProtocol, Depends(get_api_key_authenticator)],
) -> SecurityContext:
    """Dependency extracting validated caller security context and tenant ID.

    SEC-01: Derives authorization strictly from server-validated credentials. Client
    supplied X-Role headers are completely ignored to prevent privilege escalation.
    SEC-02 / SEC-48: X-Tenant-ID is honored as a target selector only for explicitly
    authorized admin contexts. Non-admin callers are strictly locked to their authenticated tenant.
    """
    if hasattr(auth, "resolve_security_context"):
        return auth.resolve_security_context(request)

    caller_id = auth.authenticate(request) or "anonymous"
    tenant_id = request.headers.get("X-Tenant-ID", "default").strip() or "default"
    return SecurityContext(
        user_id=caller_id,
        tenant_id=tenant_id,
        roles=("user",),
        is_authenticated=caller_id != "anonymous",
    )
