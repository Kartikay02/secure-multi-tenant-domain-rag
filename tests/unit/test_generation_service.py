import uuid

import pytest
from pydantic import ValidationError

from app.core.config import LLMSettings
from app.rag.context.domain import AssembledContext, ContextDocument
from app.rag.generation.factory import LLMFactory
from app.rag.generation.mock import MockLLMProvider
from app.rag.generation.openai import OpenAILLMProvider
from app.rag.generation.service import GenerationService


def _make_context(doc_title: str = "WAL Architecture") -> AssembledContext:
    doc = ContextDocument(
        citation_id=1,
        citation_label="[1]",
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        source_id="ha_manual.md",
        title=doc_title,
        page_number=3,
        score=0.95,
        content="PostgreSQL physical streaming ships write-ahead logs continuously to standby nodes.",
        token_count=15,
    )

    formatted = (
        "--- Context Document [1] ---\n"
        f"Title: {doc_title} | Source: ha_manual.md | Page: 3\n"
        "Content:\nPostgreSQL physical streaming ships write-ahead logs continuously to standby nodes.\n"
    )

    return AssembledContext(
        formatted_context=formatted,
        documents=[doc],
        citation_map={1: doc},
        total_tokens=25,
        total_chunks=1,
        truncated=False,
        dropped_chunks_count=0,
    )


@pytest.mark.asyncio
async def test_generation_service_generates_and_links_citations() -> None:
    mock_provider = MockLLMProvider()
    service = GenerationService(llm_provider=mock_provider)
    context = _make_context()

    response = await service.generate_answer(
        query="Explain streaming replication",
        context=context,
    )

    assert "[1]" in response.answer
    assert response.citations == [1]
    assert len(response.referenced_documents) == 1
    assert response.referenced_documents[0].citation_id == 1
    assert response.referenced_documents[0].title == "WAL Architecture"
    assert response.confidence_score == 0.95
    assert response.grounded is True
    assert response.insufficient_context is False


@pytest.mark.asyncio
async def test_generation_service_streaming() -> None:
    mock_provider = MockLLMProvider()
    service = GenerationService(llm_provider=mock_provider)
    context = _make_context()

    tokens: list[str] = []
    async for tok in service.stream_answer(query="Explain replication", context=context):
        tokens.append(tok)

    assert len(tokens) > 1
    assert "[1]" in "".join(tokens)


def test_llm_factory() -> None:
    # 1. Mock
    cfg_mock = LLMSettings(provider="mock", model="mock-test")
    provider_mock = LLMFactory.create_provider(cfg_mock)
    assert isinstance(provider_mock, MockLLMProvider)

    # 2. OpenAI
    cfg_openai = LLMSettings(provider="openai", model="gpt-4o-mini", api_key="test-key")
    provider_openai = LLMFactory.create_provider(cfg_openai)
    assert isinstance(provider_openai, OpenAILLMProvider)

    # 3. Invalid
    with pytest.raises(ValidationError):
        LLMSettings(provider="invalid")  # type: ignore[arg-type]
