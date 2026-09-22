"""Unit tests for tracing abstraction and InMemoryTracer."""

import pytest

from app.observability.tracer import (
    InMemoryTracer,
    NoOpTracer,
    set_trace_id,
)


@pytest.mark.asyncio
async def test_tracer_span_creation_and_attributes() -> None:
    tracer = InMemoryTracer()
    set_trace_id("test-trace-12345")

    async with tracer.span("retrieval_step", attributes={"query": "What is RAG?"}) as span:
        span.set_attribute("candidates_count", 15)
        span.set_status("OK")

    spans = tracer.recorded_spans
    assert len(spans) == 1
    s = spans[0]
    assert s.name == "retrieval_step"
    assert s.trace_id == "test-trace-12345"
    assert s.attributes["query"] == "What is RAG?"
    assert s.attributes["candidates_count"] == 15
    assert s.status == "OK"
    assert s.duration_ms >= 0.0


@pytest.mark.asyncio
async def test_tracer_nested_parent_child_hierarchy() -> None:
    tracer = InMemoryTracer()

    async with tracer.span("parent_query"):
        async with tracer.span("child_retrieval") as retr_span:
            retr_span.set_attribute("stage", "dense")
        async with tracer.span("child_rerank") as rerank_span:
            rerank_span.set_attribute("stage", "cohere")

    spans = tracer.recorded_spans
    assert len(spans) == 3

    from app.observability.tracer import SpanData

    # Concluded in order of exit: child_retrieval, child_rerank, parent_query
    child_retrieval: SpanData = next(s for s in spans if s.name == "child_retrieval")
    child_rerank: SpanData = next(s for s in spans if s.name == "child_rerank")
    parent_span: SpanData = next(s for s in spans if s.name == "parent_query")

    assert child_retrieval.parent_span_id == parent_span.span_id
    assert child_rerank.parent_span_id == parent_span.span_id
    assert parent_span.parent_span_id is None
    assert child_retrieval.trace_id == parent_span.trace_id


@pytest.mark.asyncio
async def test_tracer_exception_handling() -> None:
    tracer = InMemoryTracer()

    with pytest.raises(ValueError, match="Database disconnected"):
        async with tracer.span("failing_stage"):
            raise ValueError("Database disconnected")

    spans = tracer.recorded_spans
    assert len(spans) == 1
    s = spans[0]
    assert s.status == "ERROR"
    assert len(s.events) == 1
    assert s.events[0]["exception_type"] == "ValueError"
    assert "Database disconnected" in s.events[0]["message"]


@pytest.mark.asyncio
async def test_noop_tracer() -> None:
    tracer = NoOpTracer()
    async with tracer.span("noop_span") as span:
        span.set_attribute("key", "val")
        span.set_status("OK")
