"""Unit tests for API authentication abstraction and APIKeyAuthenticator."""

import pytest
from starlette.requests import Request

from app.api.auth import APIKeyAuthenticator, NoOpAuthenticator
from app.core.exceptions import UnauthorizedError


def _make_mock_request(headers: dict[str, str]) -> Request:
    """Helper creating a minimal Starlette Request with specified headers."""
    raw_headers = [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers.items()]
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/query",
        "headers": raw_headers,
    }
    return Request(scope)


class TestAPIKeyAuthenticator:
    """Test suite for APIKeyAuthenticator header parsing, token schemes, and timing resistance."""

    def test_auth_success_with_configured_header(self) -> None:
        authenticator = APIKeyAuthenticator(api_key="secret-token-123", header_name="X-API-Key")
        req = _make_mock_request({"X-API-Key": "secret-token-123"})
        result = authenticator.authenticate(req)
        assert result == "secret-token-123"

    def test_auth_success_with_bearer_token(self) -> None:
        authenticator = APIKeyAuthenticator(api_key="bearer-secret-456")
        req = _make_mock_request({"Authorization": "Bearer bearer-secret-456"})
        result = authenticator.authenticate(req)
        assert result == "bearer-secret-456"

    def test_auth_missing_header_raises_unauthorized(self) -> None:
        authenticator = APIKeyAuthenticator(api_key="secret-token-123")
        req = _make_mock_request({})
        with pytest.raises(UnauthorizedError) as exc_info:
            authenticator.authenticate(req)
        assert exc_info.value.status_code == 401
        assert "Missing API key" in exc_info.value.message

    def test_auth_invalid_key_raises_unauthorized(self) -> None:
        authenticator = APIKeyAuthenticator(api_key="secret-token-123")
        req = _make_mock_request({"X-API-Key": "wrong-token-999"})
        with pytest.raises(UnauthorizedError) as exc_info:
            authenticator.authenticate(req)
        assert exc_info.value.status_code == 401
        assert "Invalid API key" in exc_info.value.message

    def test_auth_disabled_when_api_key_empty(self) -> None:
        authenticator = APIKeyAuthenticator(api_key="", app_env="development")
        req = _make_mock_request({})
        result = authenticator.authenticate(req)
        assert result == "anonymous"
        ctx = authenticator.resolve_security_context(req)
        assert ctx.is_authenticated is False
        assert ctx.roles == ("user",)

    def test_auth_empty_key_fails_closed_in_production_and_staging(self) -> None:
        from app.core.exceptions import UnauthorizedError

        # Production with no keys configured must fail closed
        auth_prod = APIKeyAuthenticator(api_key="", app_env="production")
        req = _make_mock_request({})
        with pytest.raises(UnauthorizedError) as exc_info:
            auth_prod.authenticate(req)
        assert exc_info.value.status_code == 401

        # Staging with no keys configured must fail closed
        auth_staging = APIKeyAuthenticator(api_key="", app_env="staging")
        with pytest.raises(UnauthorizedError):
            auth_staging.resolve_security_context(req)

    def test_noop_authenticator(self) -> None:
        authenticator = NoOpAuthenticator(app_env="development")
        req = _make_mock_request({})
        assert authenticator.authenticate(req) == "test-user"
        ctx = authenticator.resolve_security_context(req)
        assert ctx.roles == ("user",)  # Default is standard user, not admin

    def test_noop_authenticator_forbidden_outside_dev_test(self) -> None:
        from app.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            NoOpAuthenticator(app_env="production")

        with pytest.raises(ConfigurationError):
            NoOpAuthenticator(app_env="staging")

        with pytest.raises(ConfigurationError):
            NoOpAuthenticator(app_env="uat")

    def test_tenant_api_key_derives_tenant_security_context(self) -> None:
        authenticator = APIKeyAuthenticator(
            api_key="master-admin-key",
            tenant_api_keys={"tenant-dev-key-1": "tenant-dev-1"},
        )
        req = _make_mock_request({"X-API-Key": "tenant-dev-key-1"})
        sec_ctx = authenticator.authenticate_context(req)
        assert sec_ctx.tenant_id == "tenant-dev-1"
        assert "user" in sec_ctx.roles
        assert "admin" not in sec_ctx.roles

    def test_key_rotation_accepts_secondary_active_key(self) -> None:
        authenticator = APIKeyAuthenticator(
            api_key="current-primary-key",
            rotation_api_key="incoming-secondary-key",
        )
        req1 = _make_mock_request({"X-API-Key": "current-primary-key"})
        sec_ctx1 = authenticator.authenticate_context(req1)
        assert "admin" in sec_ctx1.roles

        req2 = _make_mock_request({"X-API-Key": "incoming-secondary-key"})
        sec_ctx2 = authenticator.authenticate_context(req2)
        assert "admin" in sec_ctx2.roles

    def test_revoked_api_key_immediately_denied(self) -> None:
        authenticator = APIKeyAuthenticator(
            api_key="primary-key",
            tenant_api_keys={"leaked-key": "tenant-compromised"},
            revoked_api_keys=["leaked-key"],
        )
        req = _make_mock_request({"X-API-Key": "leaked-key"})
        with pytest.raises(UnauthorizedError) as exc_info:
            authenticator.authenticate_context(req)
        assert exc_info.value.status_code == 401
        assert "revoked" in exc_info.value.message.lower()

    def test_revoked_api_key_fingerprint_denied(self) -> None:
        import hashlib

        compromised = "compromised-secret-token"
        fp = hashlib.sha256(compromised.encode("utf-8")).hexdigest()
        authenticator = APIKeyAuthenticator(
            api_key=compromised,
            revoked_api_keys=[fp],
        )
        req = _make_mock_request({"X-API-Key": compromised})
        with pytest.raises(UnauthorizedError) as exc_info:
            authenticator.authenticate_context(req)
        assert exc_info.value.status_code == 401
        assert "revoked" in exc_info.value.message.lower()

    def test_client_cannot_spoof_admin_role_via_header(self) -> None:
        from app.core.exceptions import ForbiddenError

        authenticator = APIKeyAuthenticator(
            api_key="master-admin-key",
            tenant_api_keys={"tenant-user-key": "tenant-1"},
        )
        # Attempt 1: Role spoofing with correct tenant -> roles remain ("user",)
        req1 = _make_mock_request(
            {
                "X-API-Key": "tenant-user-key",
                "X-Role": "admin",
            }
        )
        sec_ctx = authenticator.authenticate_context(req1)
        assert "admin" not in sec_ctx.roles
        assert sec_ctx.tenant_id == "tenant-1"

        # Attempt 2: Tenant spoofing with mismatched X-Tenant-ID -> raises ForbiddenError
        req2 = _make_mock_request(
            {
                "X-API-Key": "tenant-user-key",
                "X-Tenant-ID": "other-tenant",
            }
        )
        with pytest.raises(ForbiddenError):
            authenticator.authenticate_context(req2)
