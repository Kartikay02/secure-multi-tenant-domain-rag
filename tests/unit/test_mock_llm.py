"""Unit tests for MockLLMProvider."""

import pytest

from app.core.exceptions import (
    GenerationError,
    GenerationRateLimitError,
    GenerationTimeoutError,
)
from app.rag.generation.domain import StructuredGenerationPayload
from app.rag.generation.interfaces import LLMProviderProtocol
from app.rag.generation.mock import MockLLMProvider


@pytest.mark.asyncio
async def test_mock_llm_implements_protocol() -> None:
    provider = MockLLMProvider()
    assert isinstance(provider, LLMProviderProtocol)


@pytest.mark.asyncio
async def test_mock_llm_generate_with_context() -> None:
    provider = MockLLMProvider(model_name="mock-model")
    prompt = (
        "Context Documents:\n"
        "--- Context Document [1] ---\n"
        "PostgreSQL WAL replay maintains standby replication.\n\n"
        "User Question:\n"
        "How is replication maintained?\n\n"
        "Answer the question following the grounding and citation directives."
    )

    response = await provider.generate(prompt=prompt, system_prompt="system")

    assert response.grounded is True
    assert response.insufficient_context is False
    assert response.confidence_score == 0.95
    assert response.citations == [1]
    assert "[1]" in response.answer
    assert response.model == "mock-model"
    assert response.total_tokens is not None


@pytest.mark.asyncio
async def test_mock_llm_insufficient_context_declaration() -> None:
    provider = MockLLMProvider()
    prompt = (
        "Context Documents:\n"
        "[No relevant context documents found]\n\n"
        "User Question:\n"
        "What is string theory?\n\n"
        "Answer the question."
    )

    response = await provider.generate(prompt=prompt)

    assert response.insufficient_context is True
    assert response.confidence_score == 0.0
    assert response.citations == []
    assert "I do not have sufficient information in the provided context" in response.answer


@pytest.mark.asyncio
async def test_mock_llm_structured_output_validation() -> None:
    provider = MockLLMProvider()
    prompt = "Context Documents:\n--- Context Document [1] ---\nInfo\nUser Question:\ninfo\nAnswer"

    response = await provider.generate(
        prompt=prompt,
        structured_schema=StructuredGenerationPayload,
    )

    assert "structured_payload" in response.metadata
    assert response.metadata["structured_payload"]["answer"] == response.answer
    assert response.metadata["structured_payload"]["confidence_score"] == response.confidence_score


@pytest.mark.asyncio
async def test_mock_llm_streaming() -> None:
    provider = MockLLMProvider()
    prompt = "Context Documents:\n--- Context Document [1] ---\nInfo\nUser Question:\ntest\nAnswer"

    tokens: list[str] = []
    async for token in provider.generate_stream(prompt=prompt):
        tokens.append(token)

    assert len(tokens) > 1
    reconstructed = "".join(tokens)
    assert "[1]" in reconstructed


@pytest.mark.asyncio
async def test_mock_llm_simulated_failures() -> None:
    # 1. Error
    failing = MockLLMProvider(should_fail=True)
    with pytest.raises(GenerationError) as exc_info:
        await failing.generate("test")
    assert "Mock generation failure simulated" in str(exc_info.value)

    # 2. Timeout
    timeout = MockLLMProvider(simulate_timeout=True)
    with pytest.raises(GenerationTimeoutError) as exc_info:
        await timeout.generate("test")
    assert exc_info.value.status_code == 504

    # 3. Rate limit
    rate_limit = MockLLMProvider(simulate_rate_limit=True)
    with pytest.raises(GenerationRateLimitError) as exc_info:
        await rate_limit.generate("test")
    assert exc_info.value.status_code == 429
