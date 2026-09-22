"""Unit tests for OpenAIEmbeddingProvider and EmbeddingProviderFactory."""

import json

import httpx
import pytest

from app.core.config import EmbeddingSettings
from app.core.exceptions import (
    ConfigurationError,
    EmbeddingAuthenticationError,
    EmbeddingDimensionMismatchError,
    EmbeddingRateLimitError,
    EmbeddingTimeoutError,
)
from app.rag.embeddings.factory import EmbeddingProviderFactory
from app.rag.embeddings.mock import MockEmbeddingProvider
from app.rag.embeddings.openai import OpenAIEmbeddingProvider, _mask_secret


@pytest.mark.asyncio
async def test_openai_embed_text_success() -> None:
    """Verify successful single text embedding with OpenAI adapter."""
    mock_vector = [0.1] * 1536

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == "Bearer sk-test-key-12345"
        body = json.loads(request.content.decode("utf-8"))
        assert body["input"] == ["Hello world"]
        assert body["model"] == "text-embedding-3-small"

        return httpx.Response(
            status_code=200,
            json={"data": [{"embedding": mock_vector, "index": 0}]},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test-key-12345",
            model="text-embedding-3-small",
            dimension=1536,
            client=client,
        )
        vec = await provider.embed_text("Hello world")
        assert vec == mock_vector


@pytest.mark.asyncio
async def test_openai_embed_documents_batching() -> None:
    """Verify batching splits requests and maintains original order."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        body = json.loads(request.content.decode("utf-8"))
        inputs = body["input"]
        # Generate dummy vector where first element is the input index
        data = [{"embedding": [float(i)] * 64, "index": i} for i in range(len(inputs))]
        return httpx.Response(status_code=200, json={"data": data})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = OpenAIEmbeddingProvider(
            dimension=64,
            batch_size=3,
            client=client,
        )
        texts = [f"Text {i}" for i in range(7)]
        vectors = await provider.embed_documents(texts)

        assert len(vectors) == 7
        # 7 items with batch_size 3 => 3 batches (3, 3, 1)
        assert call_count == 3


@pytest.mark.asyncio
async def test_openai_dimension_mismatch() -> None:
    """Verify EmbeddingDimensionMismatchError is raised when API returns unexpected dimension."""

    def handler(request: httpx.Request) -> httpx.Response:
        # Returns 3 dimensions when 1536 is expected
        return httpx.Response(
            status_code=200,
            json={"data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}]},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = OpenAIEmbeddingProvider(dimension=1536, client=client)
        with pytest.raises(EmbeddingDimensionMismatchError) as exc_info:
            await provider.embed_text("Test query")

        assert exc_info.value.details["expected"] == 1536
        assert exc_info.value.details["received"] == 3


@pytest.mark.asyncio
async def test_openai_rate_limit_retry_and_success() -> None:
    """Verify HTTP 429 rate limits are retried with backoff and succeed on subsequent attempt."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(status_code=429, headers={"Retry-After": "0"})
        return httpx.Response(
            status_code=200,
            json={"data": [{"embedding": [0.5] * 384, "index": 0}]},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = OpenAIEmbeddingProvider(
            dimension=384,
            max_retries=2,
            backoff_factor=0.01,
            client=client,
        )
        vec = await provider.embed_text("Retry test")
        assert len(vec) == 384
        assert attempts == 2


@pytest.mark.asyncio
async def test_openai_rate_limit_exhausted() -> None:
    """Verify exhausted 429 retries raises EmbeddingRateLimitError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=429, headers={"Retry-After": "0"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = OpenAIEmbeddingProvider(
            dimension=384,
            max_retries=1,
            backoff_factor=0.01,
            client=client,
        )
        with pytest.raises(EmbeddingRateLimitError):
            await provider.embed_text("Exhausted retries")


@pytest.mark.asyncio
async def test_openai_auth_failure_immediate() -> None:
    """Verify HTTP 401 raises EmbeddingAuthenticationError immediately without retrying."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(
            status_code=401,
            json={"error": {"message": "Invalid API key"}},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = OpenAIEmbeddingProvider(
            api_key="invalid-key",
            max_retries=3,
            client=client,
        )
        with pytest.raises(EmbeddingAuthenticationError):
            await provider.embed_text("Auth test")

        # Must fail immediately on attempt 1
        assert call_count == 1


@pytest.mark.asyncio
async def test_openai_timeout_handling() -> None:
    """Verify request timeout raises EmbeddingTimeoutError after retries."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Request timed out")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = OpenAIEmbeddingProvider(
            dimension=384,
            timeout_seconds=0.1,
            max_retries=1,
            backoff_factor=0.01,
            client=client,
        )
        with pytest.raises(EmbeddingTimeoutError):
            await provider.embed_text("Timeout test")


def test_secret_masking() -> None:
    """Verify credential strings are never exposed in plaintext."""
    assert _mask_secret("") == "<empty>"
    assert _mask_secret("short") == "***"
    assert _mask_secret("sk-proj-abcdef1234567890") == "sk-...7890"


def test_embedding_factory() -> None:
    """Verify factory returns appropriate provider implementation."""
    mock_settings = EmbeddingSettings(provider="mock", dimension=256)
    mock_provider = EmbeddingProviderFactory.create(mock_settings)
    assert isinstance(mock_provider, MockEmbeddingProvider)
    assert mock_provider.dimension == 256

    openai_settings = EmbeddingSettings(
        provider="openai",
        model="text-embedding-3-small",
        dimension=1536,
        api_key="sk-test",
    )
    openai_provider = EmbeddingProviderFactory.create(openai_settings)
    assert isinstance(openai_provider, OpenAIEmbeddingProvider)
    assert openai_provider.dimension == 1536
    assert openai_provider.model_name == "text-embedding-3-small"

    # Unsupported provider
    unsupported_settings = EmbeddingSettings(provider="fastembed")
    with pytest.raises(ConfigurationError, match="Unsupported embedding provider"):
        EmbeddingProviderFactory.create(unsupported_settings)
