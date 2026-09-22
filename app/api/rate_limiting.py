"""Rate limiting abstractions and in-memory/Redis sliding-window implementations."""

import asyncio
import hashlib
import time
from collections import defaultdict
from typing import Any, Protocol, runtime_checkable

from fastapi import Request

from app.core.config import resolve_redis_url, scrub_credentials
from app.core.exceptions import ConfigurationError, RateLimitExceededError
from app.core.logging import get_logger

logger = get_logger("app.api.rate_limiting")


@runtime_checkable
class RateLimiterProtocol(Protocol):
    """Protocol for API rate-limiting governance."""

    async def check_rate_limit(self, key: str) -> None:
        """Verify client has not exceeded request rate threshold.

        Args:
            key: Client identifier (e.g. IP address or API key).

        Raises:
            RateLimitExceededError: If request rate exceeds permitted window.
        """
        ...


class InMemoryRateLimiter(RateLimiterProtocol):
    """Sliding-window in-memory rate limiter with striped concurrency and strictly bounded memory."""

    def __init__(
        self,
        max_requests: int = 60,
        window_seconds: int = 60,
        max_keys: int = 10_000,
        num_stripes: int = 32,
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        self.num_stripes = num_stripes
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._locks = [asyncio.Lock() for _ in range(num_stripes)]
        self._capacity_lock = asyncio.Lock()

    def _get_lock(self, key: str) -> asyncio.Lock:
        stripe_idx = abs(hash(key)) % self.num_stripes
        return self._locks[stripe_idx]

    async def check_rate_limit(self, key: str) -> None:
        """Enforce sliding window rate limit for given client key with hard capacity bounding."""
        now = time.time()
        window_start = now - self.window_seconds
        lock = self._get_lock(key)

        async with lock:
            # Prune expired requests for this key
            timestamps = [t for t in self._requests.get(key, []) if t > window_start]
            if timestamps:
                self._requests[key] = timestamps
            elif key in self._requests:
                self._requests.pop(key, None)

            if len(timestamps) >= self.max_requests:
                oldest = timestamps[0]
                retry_after = int(max(1.0, (oldest + self.window_seconds) - now))
                logger.warning(
                    f"Rate limit exceeded for client '{key}': {len(timestamps)} requests in {self.window_seconds}s.",
                    extra={"client_key": key, "retry_after": retry_after},
                )
                raise RateLimitExceededError(retry_after_seconds=retry_after)

            # Enforce hard upper bound on keys: state size never exceeds max_keys
            if key not in self._requests and len(self._requests) >= self.max_keys:
                async with self._capacity_lock:
                    while len(self._requests) >= self.max_keys:
                        # 1. First evict any expired keys
                        expired_keys = [
                            k for k, v in self._requests.items() if not v or v[-1] <= window_start
                        ]
                        if expired_keys:
                            for exp_k in expired_keys:
                                self._requests.pop(exp_k, None)
                                if len(self._requests) < self.max_keys:
                                    break
                            if len(self._requests) < self.max_keys:
                                break

                        # 2. If still at capacity, evict oldest active LRU key
                        if self._requests:
                            oldest_key = min(
                                self._requests.keys(),
                                key=lambda k: self._requests[k][-1] if self._requests[k] else 0.0,
                            )
                            self._requests.pop(oldest_key, None)
                        else:
                            break

            self._requests[key].append(now)


class RedisRateLimiter(RateLimiterProtocol):
    """Distributed sliding-window rate limiter backed by Redis sorted sets.

    SEC-47 / SEC-57: Implements multi-worker sliding window in Redis.
    Fails closed if the configured Redis rate limit backend is unavailable in production.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        password: str | None = None,
        username: str | None = None,
        max_requests: int = 60,
        window_seconds: int = 60,
        client: Any = None,
        fail_closed: bool = True,
    ) -> None:
        self.redis_url = resolve_redis_url(redis_url, password=password, username=username)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._client = client
        self.fail_closed = fail_closed

    async def _get_client(self) -> Any:
        if self._client is None:
            try:
                import redis.asyncio as aioredis

                self._client = aioredis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_timeout=2.0,
                    socket_connect_timeout=2.0,
                )
            except ImportError as exc:
                raise ConfigurationError(
                    "Redis rate limiting is enabled ('SECURITY_RATE_LIMIT_BACKEND=redis'), "
                    "but the 'redis' package is not installed. Install with 'pip install redis'."
                ) from exc
        return self._client

    async def check_rate_limit(self, key: str) -> None:
        """Enforce distributed sliding-window rate limit in Redis."""
        now = time.time()
        window_start = now - self.window_seconds
        redis_key = f"rl:{key}"

        try:
            client = await self._get_client()
            async with client.pipeline(transaction=True) as pipe:
                pipe.zremrangebyscore(redis_key, 0, window_start)
                pipe.zcard(redis_key)
                pipe.zadd(redis_key, {f"{now}": now})
                pipe.expire(redis_key, self.window_seconds + 5)
                results = await pipe.execute()

            current_count = results[1]
            if current_count >= self.max_requests:
                # Remove the tentatively added score since client exceeded limit
                try:
                    await client.zrem(redis_key, f"{now}")
                except Exception:
                    pass
                retry_after = self.window_seconds
                logger.warning(
                    f"Redis rate limit exceeded for client '{key}': {current_count} requests in {self.window_seconds}s.",
                    extra={"client_key": key, "retry_after": retry_after},
                )
                raise RateLimitExceededError(retry_after_seconds=retry_after)

        except RateLimitExceededError:
            raise
        except Exception as exc:
            scrubbed_exc = scrub_credentials(str(exc))
            # SEC-57: Fail-closed in production when Redis rate limit backend is required
            if self.fail_closed:
                logger.error(f"Redis rate limiter failure: {scrubbed_exc}. Failing closed.")
                raise RateLimitExceededError(
                    retry_after_seconds=self.window_seconds,
                    message="Rate limiting service unavailable. Request rejected for security.",
                ) from None
            else:
                logger.warning(
                    f"Redis rate limiter failure: {scrubbed_exc}. Permitted under fail-open/dev policy."
                )


class NoOpRateLimiter(RateLimiterProtocol):
    """Pass-through rate limiter for unrestricted testing."""

    async def check_rate_limit(self, key: str) -> None:
        """Allow all requests unconditionally."""
        return None


def get_client_identifier(
    request: Request,
    trusted_proxies: list[str] | tuple[str, ...] | set[str] | None = None,
    authenticated_identity: str | None = None,
) -> str:
    """Extract a stable, privacy-safe, spoof-resistant client tracking key.

    Security model:
    - Authenticated requests: rate-limited by validated principal / tenant identity.
    - Unauthenticated or invalid requests: strictly rate-limited by trusted source IP.
    - Unvalidated client-supplied X-API-Key / Authorization headers are NEVER used
      as a rate-limit bucket for unauthenticated callers, preventing attackers from
      rotating fake headers to bypass rate limits.
    """
    if authenticated_identity:
        key_hash = hashlib.sha256(authenticated_identity.encode("utf-8")).hexdigest()[:16]
        return f"auth:{key_hash}"

    # Extract trusted peer socket IP
    trusted = (
        set(trusted_proxies) if trusted_proxies is not None else {"127.0.0.1", "::1", "localhost"}
    )
    peer_ip = request.client.host if (request.client and request.client.host) else "127.0.0.1"

    # Only parse X-Forwarded-For if peer is in trusted_proxies
    if peer_ip in trusted:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
            return f"ip:{client_ip}"

    return f"ip:{peer_ip}"
