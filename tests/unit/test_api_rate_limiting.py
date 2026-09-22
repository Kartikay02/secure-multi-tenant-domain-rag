"""Unit tests for rate limiting abstractions and InMemoryRateLimiter."""

import asyncio
import hashlib
from unittest.mock import MagicMock

import pytest
from starlette.requests import Request

from app.api.rate_limiting import (
    InMemoryRateLimiter,
    NoOpRateLimiter,
    RedisRateLimiter,
    get_client_identifier,
)
from app.core.exceptions import RateLimitExceededError


def _make_mock_request(headers: dict[str, str] | None = None, host: str = "127.0.0.1") -> Request:
    """Helper creating a minimal Starlette Request for rate limiting tests."""
    raw_headers = [
        (k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in (headers or {}).items()
    ]
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/query",
        "headers": raw_headers,
        "client": (host, 12345),
    }
    return Request(scope)


class TestInMemoryRateLimiter:
    """Test suite for sliding-window request counting, window eviction, and reset behavior."""

    @pytest.mark.asyncio
    async def test_allows_requests_within_limit(self) -> None:
        limiter = InMemoryRateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            await limiter.check_rate_limit("test-client")

    @pytest.mark.asyncio
    async def test_blocks_request_exceeding_threshold(self) -> None:
        limiter = InMemoryRateLimiter(max_requests=2, window_seconds=60)
        await limiter.check_rate_limit("test-client")
        await limiter.check_rate_limit("test-client")
        with pytest.raises(RateLimitExceededError) as exc_info:
            await limiter.check_rate_limit("test-client")
        assert exc_info.value.status_code == 429
        assert exc_info.value.retry_after_seconds > 0

    @pytest.mark.asyncio
    async def test_rate_limiting_tracks_clients_independently(self) -> None:
        limiter = InMemoryRateLimiter(max_requests=1, window_seconds=60)
        await limiter.check_rate_limit("client-a")
        # client-b should be allowed independently
        await limiter.check_rate_limit("client-b")

    @pytest.mark.asyncio
    async def test_noop_rate_limiter(self) -> None:
        limiter = NoOpRateLimiter()
        for _ in range(100):
            await limiter.check_rate_limit("client-unlimited")


class TestClientIdentifierResolution:
    """Test suite for extracting client tracking key from request headers and validated identity."""

    def test_unauthenticated_requests_use_ip_not_unvalidated_api_key(self) -> None:
        """Unvalidated API keys must NOT be used as client identifier for unauthenticated callers."""
        api_key = "my-secret-key-123456789"
        req = _make_mock_request(headers={"X-API-Key": api_key}, host="192.168.1.50")
        # Must fall back to IP, not create a key-based bucket
        assert get_client_identifier(req) == "ip:192.168.1.50"

    def test_rotating_invalid_keys_shares_same_ip_bucket(self) -> None:
        """Attackers rotating fake API keys from the same IP must hit the exact same IP bucket."""
        req1 = _make_mock_request(headers={"X-API-Key": "fake-key-1"}, host="203.0.113.5")
        req2 = _make_mock_request(headers={"X-API-Key": "fake-key-2"}, host="203.0.113.5")
        req3 = _make_mock_request(headers={"X-API-Key": "fake-key-3"}, host="203.0.113.5")

        assert get_client_identifier(req1) == "ip:203.0.113.5"
        assert get_client_identifier(req2) == "ip:203.0.113.5"
        assert get_client_identifier(req3) == "ip:203.0.113.5"

    def test_authenticated_identity_uses_auth_hash(self) -> None:
        """Validated credentials produce a stable, privacy-safe auth-scoped bucket."""
        req = _make_mock_request(host="192.168.1.50")
        ident = "tenant:engineering:user:usr_999"
        expected_hash = hashlib.sha256(ident.encode("utf-8")).hexdigest()[:16]
        assert get_client_identifier(req, authenticated_identity=ident) == f"auth:{expected_hash}"

    def test_different_tenants_remain_isolated(self) -> None:
        """Different authenticated tenants have distinct rate limiting buckets."""
        req = _make_mock_request(host="127.0.0.1")
        id1 = get_client_identifier(req, authenticated_identity="tenant:tenant_a:user:u1")
        id2 = get_client_identifier(req, authenticated_identity="tenant:tenant_b:user:u1")
        assert id1 != id2

    def test_prefers_forwarded_for_when_peer_is_trusted_proxy(self) -> None:
        req = _make_mock_request(
            headers={"X-Forwarded-For": "203.0.113.195, 70.41.3.18"},
            host="127.0.0.1",
        )
        assert get_client_identifier(req, trusted_proxies=["127.0.0.1"]) == "ip:203.0.113.195"

    def test_ignores_forwarded_for_when_peer_is_untrusted(self) -> None:
        req = _make_mock_request(
            headers={"X-Forwarded-For": "203.0.113.195, 70.41.3.18"},
            host="198.51.100.2",
        )
        # Untrusted proxy host should NOT be allowed to spoof X-Forwarded-For
        assert get_client_identifier(req, trusted_proxies=["127.0.0.1"]) == "ip:198.51.100.2"

    def test_falls_back_to_client_host(self) -> None:
        req = _make_mock_request(host="192.168.1.50")
        assert get_client_identifier(req) == "ip:192.168.1.50"


class TestInMemoryRateLimiterHardBounded:
    """Test suite for strict hard bounding and atomic LRU eviction (Issue 4)."""

    @pytest.mark.asyncio
    async def test_strictly_bounds_keys_at_capacity(self) -> None:
        """Ensure internal state never exceeds max_keys even with >2x unique clients."""
        limiter = InMemoryRateLimiter(max_requests=10, window_seconds=60, max_keys=10)

        # Insert 25 unique client keys
        for i in range(25):
            await limiter.check_rate_limit(f"client_{i}")

        # The state dictionary must NEVER exceed max_keys (10)
        assert len(limiter._requests) <= 10

    @pytest.mark.asyncio
    async def test_evicts_oldest_key_when_all_active(self) -> None:
        """Verify oldest LRU key is evicted when all keys are active within the window."""
        limiter = InMemoryRateLimiter(max_requests=10, window_seconds=60, max_keys=3)

        await limiter.check_rate_limit("client_1")
        await asyncio.sleep(0.01)
        await limiter.check_rate_limit("client_2")
        await asyncio.sleep(0.01)
        await limiter.check_rate_limit("client_3")

        assert len(limiter._requests) == 3
        assert "client_1" in limiter._requests

        # Insert 4th client: client_1 was oldest and must be evicted
        await asyncio.sleep(0.01)
        await limiter.check_rate_limit("client_4")

        assert len(limiter._requests) <= 3
        assert "client_1" not in limiter._requests
        assert "client_4" in limiter._requests


class TestRedisRateLimiter:
    """Test suite for Redis rate limiting and fail-closed resilience (SEC-47, SEC-57)."""

    @pytest.mark.asyncio
    async def test_redis_unreachable_fails_closed_when_configured(self) -> None:
        # Mock Redis client that fails connection/execution
        mock_redis = MagicMock()
        mock_redis.pipeline.side_effect = ConnectionError("Redis cluster unreachable")

        limiter = RedisRateLimiter(
            client=mock_redis,
            max_requests=10,
            window_seconds=60,
            fail_closed=True,
        )

        with pytest.raises(RateLimitExceededError) as exc_info:
            await limiter.check_rate_limit("client-123")
        assert exc_info.value.status_code == 429
        assert "unavailable" in exc_info.value.message.lower()

    @pytest.mark.asyncio
    async def test_redis_unreachable_allows_when_fail_closed_false(self) -> None:
        mock_redis = MagicMock()
        mock_redis.pipeline.side_effect = ConnectionError("Redis offline")

        limiter = RedisRateLimiter(
            client=mock_redis,
            max_requests=10,
            window_seconds=60,
            fail_closed=False,
        )

        # In dev/test when fail_closed=False, failure logs warning and allows request through
        await limiter.check_rate_limit("client-fallback")

    def test_resolve_redis_url_with_embedded_password(self) -> None:
        from app.core.config import resolve_redis_url

        url = resolve_redis_url("redis://:mysecret@redis-host:6379/1")
        assert url == "redis://:mysecret@redis-host:6379/1"

    def test_resolve_redis_url_merges_separate_password(self) -> None:
        from app.core.config import resolve_redis_url

        url = resolve_redis_url("redis://redis-host:6379/0", password="injected-password")
        assert url == "redis://:injected-password@redis-host:6379/0"

    def test_resolve_redis_url_with_username_and_password(self) -> None:
        from app.core.config import resolve_redis_url

        url = resolve_redis_url(
            "redis://redis-host:6379/0", username="appuser", password="secret-pass"
        )
        assert url == "redis://appuser:secret-pass@redis-host:6379/0"

    def test_resolve_redis_url_handles_special_characters_without_double_encoding(self) -> None:
        from app.core.config import resolve_redis_url

        special_pass = "p@ss:w/ord!"
        url = resolve_redis_url("redis://redis-host:6379/0", password=special_pass)
        assert "@redis-host:6379/0" in url
        # Calling it again on the already encoded URL should not corrupt or double encode
        url2 = resolve_redis_url(url)
        assert url2 == url

    def test_scrub_credentials_masks_sensitive_passwords(self) -> None:
        from app.core.config import scrub_credentials

        raw = "Error connecting to redis://:supersecret123@redis-cluster:6379/0"
        scrubbed = scrub_credentials(raw)
        assert "supersecret123" not in scrubbed
        assert "redis://:***@redis-cluster:6379/0" in scrubbed

        raw_user = "postgresql+asyncpg://postgres:dbsecret999@localhost:5432/rag_db"
        scrubbed_user = scrub_credentials(raw_user)
        assert "dbsecret999" not in scrubbed_user
        assert "postgresql+asyncpg://postgres:***@localhost:5432/rag_db" in scrubbed_user

    @pytest.mark.asyncio
    async def test_redis_limiter_scrubs_password_in_error_logging(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        mock_redis = MagicMock()
        mock_redis.pipeline.side_effect = ConnectionError(
            "Failed at redis://:secretpass99@host:6379"
        )

        limiter = RedisRateLimiter(
            redis_url="redis://localhost:6379/0",
            password="secretpass99",
            client=mock_redis,
            fail_closed=True,
        )

        with caplog.at_level(logging.ERROR):
            with pytest.raises(RateLimitExceededError):
                await limiter.check_rate_limit("test-client")

        assert "secretpass99" not in caplog.text
        assert ":***@" in caplog.text
