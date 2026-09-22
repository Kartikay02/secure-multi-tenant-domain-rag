"""Unit tests for CohereReranker adapter with mock transport."""

import json
import uuid

import httpx
import pytest

from app.core.exceptions import (
    RerankAuthenticationError,
    RerankError,
    RerankTimeoutError,
)
from app.rag.reranking.cohere import CohereReranker, _mask_secret
from app.rag.reranking.interfaces import RerankerProtocol
from app.rag.vector.domain import RetrievalResult


def _make_candidate(text: str, score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        text=text,
        score=score,
        dense_score=score,
        sparse_score=score,
        metadata={"source": "test_doc"},
    )


@pytest.mark.asyncio
async def test_cohere_reranker_implements_protocol() -> None:
    reranker = CohereReranker(api_key="test-key")
    assert isinstance(reranker, RerankerProtocol)
    await reranker.close()


@pytest.mark.asyncio
async def test_cohere_reranker_successful_call() -> None:
    cand1 = _make_candidate("Document chunk 1 content", score=0.6)
    cand2 = _make_candidate("Document chunk 2 content with high relevance", score=0.4)

    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        body = json.loads(request.content.decode())
        assert body["model"] == "rerank-v3.5"
        assert body["query"] == "test query"
        assert body["documents"] == [cand1.text, cand2.text]
        assert body["top_n"] == 2
        assert body["return_documents"] is False

        assert request.headers["authorization"] == "Bearer valid-test-key"
        assert request.headers["content-type"] == "application/json"

        response_data = {
            "id": "cohere-rerank-123",
            "results": [
                {"index": 1, "relevance_score": 0.942},
                {"index": 0, "relevance_score": 0.612},
            ],
            "meta": {"billed_units": {"search_units": 1}},
        }
        return httpx.Response(status_code=200, json=response_data)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    reranker = CohereReranker(
        api_key="valid-test-key",
        model="rerank-v3.5",
        client=client,
    )

    results = await reranker.rerank(
        query="test query",
        candidates=[cand1, cand2],
        top_k=2,
    )

    assert len(captured_requests) == 1
    assert len(results) == 2

    # cand2 was re-scored as index 1 with 0.942
    assert results[0].text == cand2.text
    assert results[0].score == 0.942
    assert results[0].rerank_score == 0.942
    assert results[0].metadata["reranker"] == "rerank-v3.5"
    assert results[0].metadata["reranker_provider"] == "cohere"
    assert results[0].dense_score == cand2.dense_score

    # cand1 was re-scored as index 0 with 0.612
    assert results[1].text == cand1.text
    assert results[1].score == 0.612
    assert results[1].rerank_score == 0.612

    await client.aclose()


@pytest.mark.asyncio
async def test_cohere_reranker_missing_api_key_raises() -> None:
    reranker = CohereReranker(api_key="")
    cand = _make_candidate("Sample text")

    with pytest.raises(RerankAuthenticationError) as exc_info:
        await reranker.rerank(query="test", candidates=[cand], top_k=1)

    assert exc_info.value.status_code == 401
    assert "API key is missing or empty" in str(exc_info.value)
    await reranker.close()


@pytest.mark.asyncio
async def test_cohere_reranker_authentication_failure_401() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=401, text="Unauthorized: invalid api token")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    reranker = CohereReranker(api_key="invalid-key", client=client, max_retries=1)

    with pytest.raises(RerankAuthenticationError) as exc_info:
        await reranker.rerank(query="test", candidates=[_make_candidate("test")], top_k=1)

    assert exc_info.value.status_code == 401
    await client.aclose()


@pytest.mark.asyncio
async def test_cohere_reranker_client_error_400() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=400, text="Bad Request: invalid model")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    reranker = CohereReranker(api_key="valid-key", client=client, max_retries=1)

    with pytest.raises(RerankError) as exc_info:
        await reranker.rerank(query="test", candidates=[_make_candidate("test")], top_k=1)

    assert "Cohere reranker client error 400" in str(exc_info.value)
    await client.aclose()


@pytest.mark.asyncio
async def test_cohere_reranker_timeout_raises_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Request timed out")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    reranker = CohereReranker(
        api_key="valid-key",
        client=client,
        timeout_seconds=2.0,
        max_retries=1,
        backoff_factor=0.01,
    )

    with pytest.raises(RerankTimeoutError) as exc_info:
        await reranker.rerank(query="test", candidates=[_make_candidate("test")], top_k=1)

    assert exc_info.value.status_code == 504
    assert exc_info.value.details["provider"] == "cohere"
    await client.aclose()


@pytest.mark.asyncio
async def test_cohere_reranker_retry_on_500_recovers() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(status_code=500, text="Internal Server Error")
        return httpx.Response(
            status_code=200,
            json={"results": [{"index": 0, "relevance_score": 0.88}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    reranker = CohereReranker(
        api_key="valid-key",
        client=client,
        max_retries=2,
        backoff_factor=0.01,
    )

    results = await reranker.rerank(
        query="test",
        candidates=[_make_candidate("test text")],
        top_k=1,
    )
    assert attempts == 2
    assert len(results) == 1
    assert results[0].rerank_score == 0.88
    await client.aclose()


@pytest.mark.asyncio
async def test_cohere_reranker_empty_candidates() -> None:
    request_sent = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_sent
        request_sent = True
        return httpx.Response(status_code=200, json={"results": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    reranker = CohereReranker(api_key="valid-key", client=client)

    results = await reranker.rerank(query="test", candidates=[], top_k=5)
    assert results == []
    assert not request_sent
    await client.aclose()


def test_cohere_reranker_secret_masking() -> None:
    assert _mask_secret("") == "<empty>"
    assert _mask_secret("short") == "***"
    assert _mask_secret("coh_1234567890abcdef") == "coh...cdef"

    reranker = CohereReranker(api_key="secret-key-12345")
    assert "secret-key" not in repr(reranker)
    assert "CohereReranker" in repr(reranker)


@pytest.mark.asyncio
async def test_cohere_reranker_context_manager() -> None:
    async with CohereReranker(api_key="test-key") as reranker:
        assert repr(reranker).startswith("CohereReranker")
