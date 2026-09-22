"""Unit tests for generation domain models and Pydantic schemas."""

import uuid

import pytest
from pydantic import ValidationError

from app.rag.context.domain import ContextDocument
from app.rag.generation.domain import (
    GeneratedResponse,
    GenerationParameters,
    StreamChunk,
    StructuredGenerationPayload,
)


def _make_context_doc() -> ContextDocument:
    return ContextDocument(
        citation_id=1,
        citation_label="[1]",
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        source_id="guide.md",
        title="Guide",
        page_number=1,
        score=0.95,
        content="PostgreSQL WAL replay maintains standby replication.",
        token_count=10,
    )


def test_generation_parameters_defaults() -> None:
    params = GenerationParameters()
    assert params.temperature == 0.0
    assert params.max_tokens == 1024
    assert params.top_p == 1.0
    assert params.stop is None
    assert params.stream is False


def test_generation_parameters_validation() -> None:
    with pytest.raises(ValidationError):
        GenerationParameters(temperature=3.0)  # max is 2.0

    with pytest.raises(ValidationError):
        GenerationParameters(max_tokens=-5)


def test_structured_generation_payload_valid() -> None:
    payload = StructuredGenerationPayload(
        answer="PostgreSQL uses write-ahead logging for durability [1].",
        citations=[1],
        confidence_score=0.98,
        grounded=True,
        insufficient_context=False,
    )
    assert payload.answer.startswith("PostgreSQL")
    assert payload.citations == [1]
    assert payload.confidence_score == 0.98
    assert payload.grounded is True
    assert payload.insufficient_context is False


def test_structured_generation_payload_validation() -> None:
    with pytest.raises(ValidationError):
        # confidence_score must be <= 1.0
        StructuredGenerationPayload(
            answer="test",
            confidence_score=1.5,
        )


def test_stream_chunk_model() -> None:
    chunk = StreamChunk(delta="token", is_complete=False)
    assert chunk.delta == "token"
    assert chunk.is_complete is False


def test_generated_response_model() -> None:
    doc = _make_context_doc()
    resp = GeneratedResponse(
        answer="Durability is guaranteed by WAL [1].",
        citations=[1],
        confidence_score=0.95,
        grounded=True,
        insufficient_context=False,
        referenced_documents=[doc],
        model="gpt-4o-mini",
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        latency_seconds=0.45,
    )

    assert resp.answer == "Durability is guaranteed by WAL [1]."
    assert resp.citations == [1]
    assert len(resp.referenced_documents) == 1
    assert resp.referenced_documents[0].citation_id == 1
    assert resp.model == "gpt-4o-mini"
    assert resp.total_tokens == 120
