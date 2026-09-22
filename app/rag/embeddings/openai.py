"""Production-ready OpenAI-compatible hosted embedding provider adapter."""

import asyncio
import random
from typing import Any

import httpx

from app.core.exceptions import (
    EmbeddingAuthenticationError,
    EmbeddingDimensionMismatchError,
    EmbeddingError,
    EmbeddingRateLimitError,
    EmbeddingTimeoutError,
)
from app.core.logging import get_logger
from app.rag.embeddings.interfaces import EmbeddingProtocol

logger = get_logger("app.rag.embeddings.openai")

TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


def _mask_secret(secret: str) -> str:
    """Mask sensitive credentials for safe logging."""
    if not secret:
        return "<empty>"
    if len(secret) <= 8:
        return "***"
    return f"{secret[:3]}...{secret[-4:]}"


class OpenAIEmbeddingProvider(EmbeddingProtocol):
    """Production-grade HTTP adapter for OpenAI-compatible embedding endpoints.

    Works with OpenAI, Azure OpenAI, vLLM, Ollama, Together AI, and TEI.
    Enforces request timeouts, bounded exponential backoff with jitter,
    automatic document batching, dimension validation, and safe credential logging.
    """

    def __init__(
        self,
        api_key: str = "",
        api_base: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
        batch_size: int = 32,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        max_backoff_seconds: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")

        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self._model = model
        self._dimension = dimension
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.max_backoff_seconds = max_backoff_seconds

        self._external_client = client is not None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout_seconds),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        )

        logger.info(
            "Initialized OpenAIEmbeddingProvider",
            extra={
                "model": self._model,
                "dimension": self._dimension,
                "api_base": self._api_base,
                "api_key_masked": _mask_secret(self._api_key),
                "batch_size": self.batch_size,
                "timeout_seconds": self.timeout_seconds,
                "max_retries": self.max_retries,
            },
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model

    async def close(self) -> None:
        """Close the underlying HTTP client if internally managed."""
        if not self._external_client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> "OpenAIEmbeddingProvider":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    def _build_headers(self) -> dict[str, str]:
        """Build request headers with authorization token."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _validate_vector_dimension(self, vector: list[float], index: int) -> None:
        """Verify vector length equals expected dimension."""
        if len(vector) != self._dimension:
            logger.error(
                f"Vector at index {index} has dimension {len(vector)}, expected {self._dimension}",
                extra={"expected": self._dimension, "received": len(vector), "model": self._model},
            )
            raise EmbeddingDimensionMismatchError(
                expected=self._dimension,
                received=len(vector),
                model=self._model,
            )

    async def _execute_batch_with_retry(self, batch_texts: list[str]) -> list[list[float]]:
        """Execute a single batch request with bounded exponential backoff and jitter."""
        endpoint = f"{self._api_base}/embeddings"
        payload: dict[str, Any] = {
            "input": batch_texts,
            "model": self._model,
        }

        # Some providers support dimension parameter (e.g. OpenAI v3)
        if "text-embedding-3" in self._model:
            payload["dimensions"] = self._dimension

        attempt = 0
        while True:
            try:
                logger.debug(
                    f"Sending embedding request attempt {attempt + 1}/{self.max_retries + 1} for {len(batch_texts)} texts"
                )
                response = await self._client.post(
                    endpoint,
                    json=payload,
                    headers=self._build_headers(),
                )

                # 1. Non-retryable Authentication Errors (401/403)
                if response.status_code in (401, 403):
                    logger.error(
                        f"Authentication failed with status {response.status_code}",
                        extra={"status_code": response.status_code, "model": self._model},
                    )
                    raise EmbeddingAuthenticationError(
                        provider="openai",
                        reason=f"HTTP {response.status_code}: {response.text}",
                    )

                # 2. Check for transient errors to retry
                if response.status_code in TRANSIENT_STATUS_CODES:
                    is_rate_limit = response.status_code == 429
                    if attempt >= self.max_retries:
                        if is_rate_limit:
                            raise EmbeddingRateLimitError(provider="openai")
                        raise EmbeddingError(
                            f"Embedding request failed with HTTP {response.status_code}: {response.text}",
                            details={"status_code": response.status_code, "model": self._model},
                        )

                    # Determine backoff delay
                    retry_after = response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        delay = min(self.max_backoff_seconds, float(retry_after))
                    else:
                        base_backoff = self.backoff_factor * (2**attempt)
                        jitter = random.uniform(0.0, 0.5)
                        delay = min(self.max_backoff_seconds, base_backoff + jitter)

                    logger.warning(
                        f"Transient error HTTP {response.status_code} received. Retrying in {delay:.2f}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue

                # 3. Check for other HTTP client/server errors (e.g. 400 Bad Request)
                if response.is_error:
                    logger.error(
                        f"Embedding request failed with status {response.status_code}: {response.text}"
                    )
                    raise EmbeddingError(
                        f"Embedding API error {response.status_code}: {response.text}",
                        details={"status_code": response.status_code, "model": self._model},
                    )

                # 4. Parse response JSON
                data = response.json()
                items = data.get("data", [])
                if not items:
                    raise EmbeddingError("Embedding provider returned empty data array.")

                # Sort by index if present to guarantee ordering
                sorted_items = sorted(items, key=lambda x: x.get("index", 0))
                vectors: list[list[float]] = []

                for idx, item in enumerate(sorted_items):
                    vec = item.get("embedding", [])
                    self._validate_vector_dimension(vec, idx)
                    vectors.append(vec)

                return vectors

            except (httpx.TimeoutException, httpx.NetworkError) as net_err:
                is_timeout = isinstance(net_err, httpx.TimeoutException)
                if attempt >= self.max_retries:
                    if is_timeout:
                        raise EmbeddingTimeoutError(
                            provider="openai",
                            timeout_seconds=self.timeout_seconds,
                        ) from net_err
                    raise EmbeddingError(
                        f"Network error during embedding request: {net_err}"
                    ) from net_err

                base_backoff = self.backoff_factor * (2**attempt)
                jitter = random.uniform(0.0, 0.5)
                delay = min(self.max_backoff_seconds, base_backoff + jitter)

                err_type = "Timeout" if is_timeout else "NetworkError"
                logger.warning(
                    f"{err_type} encountered: {net_err}. Retrying in {delay:.2f}s (attempt {attempt + 1}/{self.max_retries})"
                )
                await asyncio.sleep(delay)
                attempt += 1

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding vector for a single query or text snippet."""
        if not text or not text.strip():
            # Return zero-vector for empty string to avoid API errors
            return [0.0] * self._dimension

        vectors = await self._execute_batch_with_retry([text])
        return vectors[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of document texts in batches."""
        if not texts:
            return []

        all_vectors: list[list[float]] = []

        # Split into batches respecting configured batch_size
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_vectors = await self._execute_batch_with_retry(batch)
            all_vectors.extend(batch_vectors)

        return all_vectors
