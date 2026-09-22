"""Unit tests for prompt formatting and RAG grounding directives."""

import uuid

from app.rag.context.domain import AssembledContext, ContextDocument
from app.rag.generation.interfaces import PromptBuilderProtocol
from app.rag.generation.prompt import DEFAULT_RAG_SYSTEM_PROMPT, PromptBuilder


def _make_context(has_chunks: bool = True) -> AssembledContext:
    if not has_chunks:
        return AssembledContext(
            formatted_context="",
            documents=[],
            citation_map={},
            total_tokens=0,
            total_chunks=0,
            truncated=False,
            dropped_chunks_count=0,
        )

    doc = ContextDocument(
        citation_id=1,
        citation_label="[1]",
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        source_id="ha_guide.md",
        title="HA Guide",
        page_number=2,
        score=0.92,
        content="Streaming replication sends WAL records continuously to standby nodes.",
        token_count=15,
    )

    formatted = (
        "--- Context Document [1] ---\n"
        "Title: HA Guide | Source: ha_guide.md | Page: 2\n"
        "Content:\nStreaming replication sends WAL records continuously to standby nodes.\n"
    )

    return AssembledContext(
        formatted_context=formatted,
        documents=[doc],
        citation_map={1: doc},
        total_tokens=30,
        total_chunks=1,
        truncated=False,
        dropped_chunks_count=0,
    )


def test_prompt_builder_implements_protocol() -> None:
    builder = PromptBuilder()
    assert isinstance(builder, PromptBuilderProtocol)


def test_rag_system_prompt_contains_critical_directives() -> None:
    # 1. Grounding directive
    assert "Strict Context Adherence" in DEFAULT_RAG_SYSTEM_PROMPT
    assert "Never assume, infer, extrapolate" in DEFAULT_RAG_SYSTEM_PROMPT

    # 2. Source citation directive
    assert "Source Citations" in DEFAULT_RAG_SYSTEM_PROMPT
    assert "[1]" in DEFAULT_RAG_SYSTEM_PROMPT

    # 3. Unsupported claims directive
    assert "Prevent Unsupported Claims" in DEFAULT_RAG_SYSTEM_PROMPT

    # 4. Explicit uncertainty directive
    assert "Explicit Uncertainty" in DEFAULT_RAG_SYSTEM_PROMPT
    assert (
        "I do not have sufficient information in the provided context to answer this question."
        in DEFAULT_RAG_SYSTEM_PROMPT
    )


def test_prompt_builder_with_context() -> None:
    builder = PromptBuilder()
    context = _make_context(has_chunks=True)
    system_prompt, user_prompt = builder.build_prompt(
        query="How does streaming replication work?",
        context=context,
    )

    assert system_prompt == DEFAULT_RAG_SYSTEM_PROMPT
    assert "Context Documents:" in user_prompt
    assert "--- Context Document [1] ---" in user_prompt
    assert "User Question:\nHow does streaming replication work?" in user_prompt


def test_prompt_builder_without_context() -> None:
    builder = PromptBuilder()
    context = _make_context(has_chunks=False)
    system_prompt, user_prompt = builder.build_prompt(
        query="Explain quantum computing.",
        context=context,
    )

    assert "[No relevant context documents found]" in user_prompt
    assert "Explain quantum computing." in user_prompt


def test_prompt_builder_custom_system_prompt() -> None:
    custom_sys = "Custom system prompt for domain RAG."
    builder = PromptBuilder(system_prompt=custom_sys)
    context = _make_context(has_chunks=True)
    system_prompt, _ = builder.build_prompt(query="query", context=context)

    assert system_prompt == custom_sys
