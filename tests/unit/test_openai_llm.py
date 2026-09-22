"""Unit tests for OpenAILLMProvider adapter using mock transport."""

import json

import httpx
import pytest

from app.core.exceptions import (
    GenerationAuthenticationError,
    GenerationTimeoutError,
)
from app.rag.generation.domain import GenerationParameters
from app.rag.generation.interfaces import LLMProviderProtocol
from app.rag.generation.openai import OpenAILLMProvider, _mask_secret


@pytest.mark.asyncio
async def test_openai_llm_implements_protocol() -> None:
    provider = OpenAILLMProvider(api_key="test-key")
    assert isinstance(provider, LLMProviderProtocol)
    await provider.close()


@pytest.mark.asyncio
async def test_openai_llm_successful_generation() -> None:
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        assert request.headers["authorization"] == "Bearer valid-openai-key"
        body = json.loads(request.content.decode())
        assert body["model"] == "gpt-4o-mini"
        assert body["temperature"] == 0.0
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][1]["role"] == "user"
        assert "response_format" in body

        structured_json = json.dumps(
            {
                "answer": "WAL streaming maintains standby nodes [1].",
                "citations": [1],
                "confidence_score": 0.98,
                "grounded": True,
                "insufficient_context": False,
            }
        )
        resp_data = {
            "id": "chatcmpl-test",
            "model": "gpt-4o-mini",
            "choices": [{"message": {"content": structured_json}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 20, "total_tokens": 140},
        }
        return httpx.Response(status_code=200, json=resp_data)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAILLMProvider(
        api_key="valid-openai-key",
        model="gpt-4o-mini",
        client=client,
    )

    response = await provider.generate(
        prompt="User question",
        system_prompt="System instructions",
        parameters=GenerationParameters(temperature=0.0),
    )

    assert len(captured_requests) == 1
    assert response.answer == "WAL streaming maintains standby nodes [1]."
    assert response.citations == [1]
    assert response.confidence_score == 0.98
    assert response.grounded is True
    assert response.insufficient_context is False
    assert response.total_tokens == 140
    await client.aclose()


@pytest.mark.asyncio
async def test_openai_llm_markdown_fenced_json_fallback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raw_json = json.dumps(
            {
                "answer": "Answer from markdown block [1].",
                "citations": [1],
                "confidence_score": 0.9,
                "grounded": True,
                "insufficient_context": False,
            }
        )
        fenced_content = f"```json\n{raw_json}\n```"
        resp_data = {
            "choices": [{"message": {"content": fenced_content}}],
            "usage": {"total_tokens": 100},
        }
        return httpx.Response(status_code=200, json=resp_data)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAILLMProvider(api_key="valid-key", client=client)

    response = await provider.generate(prompt="prompt")
    assert response.answer == "Answer from markdown block [1]."
    assert response.citations == [1]
    await client.aclose()


@pytest.mark.asyncio
async def test_openai_llm_missing_api_key() -> None:
    provider = OpenAILLMProvider(api_key="")
    with pytest.raises(GenerationAuthenticationError) as exc_info:
        await provider.generate("test")
    assert exc_info.value.status_code == 401
    assert "API key is missing" in str(exc_info.value)
    await provider.close()


@pytest.mark.asyncio
async def test_openai_llm_auth_error_401() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=401, text="Unauthorized: Invalid key")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAILLMProvider(api_key="bad-key", client=client, max_retries=1)

    with pytest.raises(GenerationAuthenticationError) as exc_info:
        await provider.generate("test")
    assert exc_info.value.status_code == 401
    await client.aclose()


@pytest.mark.asyncio
async def test_openai_llm_timeout_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Request timed out")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAILLMProvider(
        api_key="key",
        client=client,
        timeout_seconds=2.0,
        max_retries=1,
        backoff_factor=0.01,
    )

    with pytest.raises(GenerationTimeoutError) as exc_info:
        await provider.generate("test")
    assert exc_info.value.status_code == 504
    await client.aclose()


@pytest.mark.asyncio
async def test_openai_llm_retry_on_429_recovers() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                status_code=429, headers={"Retry-After": "0.01"}, text="Rate limited"
            )

        res_json = json.dumps(
            {
                "answer": "Recovered answer [1].",
                "citations": [1],
                "confidence_score": 1.0,
                "grounded": True,
                "insufficient_context": False,
            }
        )
        return httpx.Response(
            status_code=200, json={"choices": [{"message": {"content": res_json}}]}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAILLMProvider(
        api_key="key",
        client=client,
        max_retries=2,
        backoff_factor=0.01,
    )

    response = await provider.generate("test")
    assert attempts == 2
    assert response.answer == "Recovered answer [1]."
    await client.aclose()


@pytest.mark.asyncio
async def test_openai_llm_streaming() -> None:
    sse_body = (
        "data: " + json.dumps({"choices": [{"delta": {"content": "Hello"}}]}) + "\n\n"
        "data: " + json.dumps({"choices": [{"delta": {"content": " world"}}]}) + "\n\n"
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, text=sse_body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAILLMProvider(api_key="key", client=client)

    tokens: list[str] = []
    async for tok in provider.generate_stream("test"):
        tokens.append(tok)

    assert "".join(tokens) == "Hello world"
    await client.aclose()


def test_openai_llm_secret_masking() -> None:
    assert _mask_secret("") == "<empty>"
    assert _mask_secret("12345") == "***"
    assert _mask_secret("sk-1234567890abcdef") == "sk-...cdef"

    provider = OpenAILLMProvider(api_key="sk-secret-key-12345")
    assert "sk-secret-key" not in repr(provider)
    assert "OpenAILLMProvider" in repr(provider)
