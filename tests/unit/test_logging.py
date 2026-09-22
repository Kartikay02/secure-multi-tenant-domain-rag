"""Unit tests for structured logging and correlation ID tracking."""

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.logging import (
    ConsoleLogFormatter,
    JSONLogFormatter,
    get_correlation_id,
    set_correlation_id,
)
from app.rag.context.domain import AssembledContext
from app.rag.generation.domain import GeneratedResponse
from app.services.rag_service import RAGOrchestratorService


def test_correlation_id_context() -> None:
    """Verify set_correlation_id updates the context variable."""
    set_correlation_id("test-corr-12345")
    assert get_correlation_id() == "test-corr-12345"


def test_json_log_formatter() -> None:
    """Verify JSONLogFormatter emits valid JSON with expected keys."""
    set_correlation_id("corr-json-test")
    formatter = JSONLogFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=42,
        msg="Sample log message for testing",
        args=(),
        exc_info=None,
    )

    output = formatter.format(record)
    parsed = json.loads(output)

    assert parsed["level"] == "INFO"
    assert parsed["message"] == "Sample log message for testing"
    assert parsed["logger"] == "test_logger"
    assert parsed["correlation_id"] == "corr-json-test"
    assert "timestamp" in parsed
    assert parsed["line"] == 42


def test_console_log_formatter() -> None:
    """Verify ConsoleLogFormatter generates readable string containing correlation ID."""
    set_correlation_id("corr-console-test")
    formatter = ConsoleLogFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.WARNING,
        pathname="test.py",
        lineno=10,
        msg="Console warning message",
        args=(),
        exc_info=None,
    )

    output = formatter.format(record)
    assert "[corr-console-test]" in output
    assert "WARNING" in output
    assert "Console warning message" in output


@pytest.mark.asyncio
async def test_query_pii_scrubbed_from_orchestrator_and_retrieval_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify that sensitive query text does not leak into logs across the pipeline."""
    sensitive_query = "CONFIDENTIAL_PATIENT_SSN_987-65-4321_MEDICAL_DATA"

    mock_retriever = MagicMock()
    mock_retriever.retrieve = AsyncMock(return_value=[])

    empty_ctx = AssembledContext(
        formatted_context="",
        documents=[],
        citation_map={},
        total_tokens=0,
        total_chunks=0,
        truncated=False,
        dropped_chunks_count=0,
    )
    mock_builder = MagicMock()
    mock_builder.build_context = MagicMock(return_value=empty_ctx)

    mock_gen = MagicMock()
    mock_gen.generate_answer = AsyncMock(
        return_value=GeneratedResponse(
            answer="No data available.",
            citations=[],
            insufficient_context=True,
        )
    )

    orchestrator = RAGOrchestratorService(
        retriever=mock_retriever,
        context_builder=mock_builder,
        generation_service=mock_gen,
    )

    caplog.set_level(logging.DEBUG)
    await orchestrator.execute(query=sensitive_query)

    # Raw sensitive query text must NOT appear in any log output
    assert sensitive_query not in caplog.text
    assert "CONFIDENTIAL_PATIENT_SSN" not in caplog.text
    assert "987-65-4321" not in caplog.text
