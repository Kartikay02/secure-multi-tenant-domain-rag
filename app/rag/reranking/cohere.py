"""Production-grade Cohere Rerank API adapter with retry, backoff, and timeouts."""

import asyncio
import random
from collections.abc import Sequence
from typing import Any

import httpx

from app.core.exceptions import (
    RerankAuthenticationError,
    RerankError,
    RerankTimeoutError,
)
from app.core.logging import get_logger
from app.rag.reranking.interfaces import RerankerProtocol
from app.rag.vector.domain import RetrievalResult

logger = get_logger("app.rag.reranking.cohere")

TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


def _mask_secret(secret: str) -> str:
    """Mask sensitive API keys for safe structured logging."""
    if not secret:
        return "<empty>"
    if len(secret) <= 8:
        return "***"
    return f"{secret[:3]}...{secret[-4:]}"


class CohereReranker(RerankerProtocol):
    """Production cross-encoder reranker adapter using Cohere's /v1/rerank API.

    Features:
    - Bounded exponential backoff with jitter on 429/5xx transient errors
    - Configurable per-request timeouts
    - Safe secret masking in logs and repr
    - Maps upstream relevance scores to normalized RetrievalResult objects
    - Graceful connection pooling with optional external HTTP client
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "rerank-v3.5",
        api_base: str = "https://api.cohere.com/v1",
        timeout_seconds: float = 10.0,
        max_retries: int = 2,
        backoff_factor: float = 0.5,
        max_backoff_seconds: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {max_retries}")

        self._api_key = api_key.strip()
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.max_backoff_seconds = max_backoff_seconds

        self._external_client = client is not None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout_seconds),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=25),
        )

        logger.info(
            "Initialized CohereReranker",
            extra={
                "model": self.model,
                "api_base": self.api_base,
                "api_key_masked": _mask_secret(self._api_key),
                "timeout_seconds": self.timeout_seconds,
                "max_retries": self.max_retries,
            },
        )

    def __repr__(self) -> str:
        return f"CohereReranker(model={self.model!r}, api_base={self.api_base!r})"

    async def close(self) -> None:
        """Close the internal HTTP client if not externally injected."""
        if not self._external_client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> "CohereReranker":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Rerank candidate chunks using Cohere's /v1/rerank endpoint."""
        if not candidates:
            return []

        if not self._api_key:
            logger.error("Cohere API key missing for reranking call")
            raise RerankAuthenticationError(
                provider="cohere",
                reason="API key is missing or empty. Set COHERE_API_KEY.",
            )

        endpoint = f"{self.api_base}/rerank"
        payload = {
            "model": self.model,
            "query": query,
            "documents": [c.text for c in candidates],
            "top_n": min(top_k, len(candidates)),
            "return_documents": False,
        }

        attempt = 0
        while True:
            try:
                logger.debug(
                    f"Sending Cohere rerank attempt {attempt + 1}/{self.max_retries + 1} "
                    f"for {len(candidates)} candidates"
                )
                response = await self._client.post(
                    endpoint,
                    json=payload,
                    headers=self._build_headers(),
                )

                # 1. Non-retryable authentication failure
                if response.status_code in (401, 403):
                    logger.error(f"Cohere authentication failed with status {response.status_code}")
                    raise RerankAuthenticationError(
                        provider="cohere",
                        reason=f"HTTP {response.status_code}: {response.text}",
                    )

                # 2. Transient status codes for retry
                if response.status_code in TRANSIENT_STATUS_CODES:
                    if attempt >= self.max_retries:
                        raise RerankError(
                            f"Cohere reranker transient failure after {self.max_retries} retries: HTTP {response.status_code}",
                            details={"status_code": response.status_code, "model": self.model},
                        )

                    retry_after = response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        delay = min(self.max_backoff_seconds, float(retry_after))
                    else:
                        base = self.backoff_factor * (2**attempt)
                        jitter = random.uniform(0.0, 0.3)
                        delay = min(self.max_backoff_seconds, base + jitter)

                    logger.warning(
                        f"Transient HTTP {response.status_code} from Cohere. Retrying in {delay:.2f}s "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue

                # 3. Non-transient client errors
                if response.is_error:
                    logger.error(
                        f"Cohere rerank request failed with HTTP {response.status_code}: {response.text}"
                    )
                    raise RerankError(
                        f"Cohere reranker client error {response.status_code}: {response.text}",
                        details={"status_code": response.status_code, "model": self.model},
                    )

                # 4. Parse results
                data = response.json()
                raw_results = data.get("results", [])

                reranked: list[RetrievalResult] = []
                for item in raw_results:
                    idx = item.get("index")
                    if idx is None or not (0 <= idx < len(candidates)):
                        continue
                    relevance_score = float(item.get("relevance_score", 0.0))
                    orig = candidates[idx]

                    updated_metadata = dict(orig.metadata)
                    updated_metadata["reranker"] = self.model
                    updated_metadata["reranker_provider"] = "cohere"

                    reranked.append(
                        RetrievalResult(
                            chunk_id=orig.chunk_id,
                            document_id=orig.document_id,
                            text=orig.text,
                            score=relevance_score,
                            dense_score=orig.dense_score,
                            sparse_score=orig.sparse_score,
                            rerank_score=relevance_score,
                            metadata=updated_metadata,
                        )
                    )

                # Cohere normally returns results sorted by relevance_score descending,
                # but sort explicitly to guarantee contract invariant.
                reranked.sort(
                    key=lambda r: r.rerank_score if r.rerank_score is not None else 0.0,
                    reverse=True,
                )
                return reranked[:top_k]

            except (httpx.TimeoutException, httpx.NetworkError) as net_err:
                is_timeout = isinstance(net_err, httpx.TimeoutException)
                if attempt >= self.max_retries:
                    if is_timeout:
                        raise RerankTimeoutError(
                            provider="cohere",
                            timeout_seconds=self.timeout_seconds,
                        ) from net_err
                    raise RerankError(
                        f"Network error during Cohere rerank request: {net_err}"
                    ) from net_err

                base = self.backoff_factor * (2**attempt)
                jitter = random.uniform(0.0, 0.3)
                delay = min(self.max_backoff_seconds, base + jitter)

                logger.warning(
                    f"Network error calling Cohere ({net_err}). Retrying in {delay:.2f}s "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
                await asyncio.sleep(delay)
                attempt += 1
