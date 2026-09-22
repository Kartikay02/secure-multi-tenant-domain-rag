"""API authentication abstractions and API-key verification."""

import hashlib
import secrets
from typing import Protocol, runtime_checkable

from fastapi import Request

from app.core.exceptions import ConfigurationError, ForbiddenError, UnauthorizedError
from app.core.logging import get_logger
from app.security.authorization import SecurityContext

logger = get_logger("app.api.auth")


@runtime_checkable
class AuthenticationProviderProtocol(Protocol):
    """Protocol defining caller authentication for protected API endpoints."""

    def authenticate(self, request: Request) -> str | None:
        """Authenticate an incoming HTTP request.

        Args:
            request: The incoming Starlette/FastAPI request.

        Returns:
            Authenticated identity or token string.

        Raises:
            UnauthorizedError: If credentials are missing or invalid.
        """
        ...

    def resolve_security_context(self, request: Request) -> SecurityContext:
        """Derive validated SecurityContext from incoming request credentials."""
        ...


class APIKeyAuthenticator(AuthenticationProviderProtocol):
    """Production API key authenticator supporting custom header, Bearer tokens,
    multi-tenant key mapping, immediate key revocation, and dual-key rotation.
    """

    def __init__(
        self,
        api_key: str = "",
        header_name: str = "X-API-Key",
        tenant_api_keys: dict[str, str] | None = None,
        revoked_api_keys: list[str] | set[str] | None = None,
        rotation_api_key: str = "",
        app_env: str | None = None,
    ) -> None:
        import os

        self.api_key = api_key.strip()
        self.header_name = header_name
        self.tenant_api_keys = tenant_api_keys or {}
        self.revoked_api_keys = set(revoked_api_keys or [])
        self.rotation_api_key = rotation_api_key.strip()
        self.app_env = (app_env or os.environ.get("APP_ENV", "development")).lower()

    def _extract_key(self, request: Request) -> str | None:
        """Extract API key from header or Bearer authorization."""
        provided_key = request.headers.get(self.header_name)
        if not provided_key:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                provided_key = auth_header[7:].strip()
        return provided_key

    @staticmethod
    def _get_fingerprint(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def resolve_security_context(self, request: Request) -> SecurityContext:
        """Derive validated SecurityContext enforcing tenant locking and role derivation."""
        # If no keys are configured anywhere:
        if not self.api_key and not self.tenant_api_keys and not self.rotation_api_key:
            if self.app_env not in ("development", "test"):
                logger.error(
                    f"Authentication error: No API keys configured in '{self.app_env}' environment. Failing closed."
                )
                raise UnauthorizedError(
                    f"Authentication required: API keys are not configured for '{self.app_env}' environment."
                )
            # Dev/Test fallback: anonymous caller is unauthenticated
            target_tenant = request.headers.get("X-Tenant-ID", "default").strip() or "default"
            return SecurityContext(
                user_id="anonymous",
                tenant_id=target_tenant,
                roles=("user",),
                is_authenticated=False,
            )

        provided_key = self._extract_key(request)
        if not provided_key:
            logger.warning("Request rejected: missing API key authentication.")
            raise UnauthorizedError(
                f"Missing API key. Please provide credentials via '{self.header_name}' header "
                f"or 'Authorization: Bearer <key>'."
            )

        fp = self._get_fingerprint(provided_key)

        # SEC-58: Immediate revocation check
        if provided_key in self.revoked_api_keys or fp in self.revoked_api_keys:
            logger.warning("Request rejected: API key has been revoked.", extra={"key_fp": fp[:8]})
            raise UnauthorizedError("API key has been revoked.")

        # Master key and rotation key validation (grants admin privileges)
        is_master = bool(self.api_key and secrets.compare_digest(provided_key, self.api_key))
        is_rotation = bool(
            self.rotation_api_key and secrets.compare_digest(provided_key, self.rotation_api_key)
        )
        if is_master or is_rotation:
            # SEC-02 / SEC-48: Admin may specify target tenant selector via X-Tenant-ID
            target_tenant = request.headers.get("X-Tenant-ID", "default").strip() or "default"
            return SecurityContext(
                user_id=f"admin:{fp[:8]}",
                tenant_id=target_tenant,
                roles=("user", "admin"),
                is_authenticated=True,
            )

        # SEC-48 / Issue 6: Strict API-key -> Tenant mapping (never accept tenant_id as a key)
        matched_tenant: str | None = None
        for key, tenant in self.tenant_api_keys.items():
            if secrets.compare_digest(provided_key, key):
                matched_tenant = tenant
                break

        if matched_tenant is not None:
            # SEC-02 / SEC-48: Non-admin caller cannot override assigned tenant
            requested_tenant = request.headers.get("X-Tenant-ID", "").strip()
            if requested_tenant and requested_tenant != matched_tenant:
                logger.warning(
                    f"Tenant mismatch: caller key bound to '{matched_tenant}' requested '{requested_tenant}'."
                )
                raise ForbiddenError(
                    f"API key is restricted to tenant '{matched_tenant}', cannot access tenant '{requested_tenant}'."
                )
            return SecurityContext(
                user_id=f"user:{fp[:8]}",
                tenant_id=matched_tenant,
                roles=("user",),
                is_authenticated=True,
            )

        logger.warning("Request rejected: invalid API key provided.", extra={"key_fp": fp[:8]})
        raise UnauthorizedError("Invalid API key.")

    # Convenience alias for resolve_security_context
    authenticate_context = resolve_security_context

    def authenticate(self, request: Request) -> str | None:
        """Verify API key using constant-time string comparison."""
        ctx = self.resolve_security_context(request)
        provided_key = self._extract_key(request)
        if provided_key:
            return provided_key
        if not ctx.is_authenticated:
            return "anonymous"
        return ctx.user_id


class NoOpAuthenticator(AuthenticationProviderProtocol):
    """Pass-through authenticator strictly for testing and local development (SEC-11 / Issue 7).

    Raises ConfigurationError immediately if initialized or executed outside development/test environments.
    """

    def __init__(
        self,
        app_env: str | None = None,
        roles: tuple[str, ...] = ("user",),
        user_id: str = "test-user",
    ) -> None:
        import os

        self.app_env = (app_env or os.environ.get("APP_ENV", "development")).lower()
        if self.app_env not in ("development", "test"):
            raise ConfigurationError(
                f"NoOpAuthenticator is strictly forbidden outside development/test environments (got '{self.app_env}')."
            )
        self.roles = roles
        self.user_id = user_id

    def authenticate(self, request: Request) -> str | None:
        """Allow requests without authentication only in non-production environments."""
        if self.app_env not in ("development", "test"):
            raise ConfigurationError(
                f"NoOpAuthenticator is strictly forbidden outside development/test environments (got '{self.app_env}')."
            )
        return self.user_id

    def resolve_security_context(self, request: Request) -> SecurityContext:
        """Derive test security context strictly in non-production environments."""
        if self.app_env not in ("development", "test"):
            raise ConfigurationError(
                f"NoOpAuthenticator is strictly forbidden outside development/test environments (got '{self.app_env}')."
            )
        target_tenant = request.headers.get("X-Tenant-ID", "default").strip() or "default"
        return SecurityContext(
            user_id=self.user_id,
            tenant_id=target_tenant,
            roles=self.roles,
            is_authenticated=True,
        )
