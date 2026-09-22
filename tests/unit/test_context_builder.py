"""Unit tests for ContextBuilder."""

import uuid

from app.rag.context.builder import ContextBuilder
from app.rag.context.domain import AssembledContext
from app.rag.context.formatting import ContextFormatter
from app.rag.context.interfaces import ContextBuilderProtocol
from app.rag.vector.domain import RetrievalResult


def _make_chunk(
    text: str,
    score: float = 0.5,
    page: int | str | None = None,
    doc_name: str = "TestDoc",
    source: str = "doc.txt",
) -> RetrievalResult:
    meta: dict[str, object] = {
        "document_name": doc_name,
        "source": source,
    }
    if page is not None:
        meta["page_number"] = page

    return RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        text=text,
        score=score,
        dense_score=score,
        sparse_score=score,
        rerank_score=score,
        metadata=meta,
    )


def test_context_builder_implements_protocol() -> None:
    builder = ContextBuilder()
    assert isinstance(builder, ContextBuilderProtocol)


def test_context_builder_basic_assembly_and_citations() -> None:
    c1 = _make_chunk("Section 1 about PostgreSQL transactions.", score=0.9, page=1)
    c2 = _make_chunk("Section 2 about Raft consensus.", score=0.8, page=2)

    builder = ContextBuilder()
    assembled = builder.build_context(query="PostgreSQL Raft", candidates=[c1, c2])

    assert isinstance(assembled, AssembledContext)
    assert len(assembled.documents) == 2
    assert assembled.total_chunks == 2
    assert assembled.truncated is False
    assert assembled.dropped_chunks_count == 0

    # Verify sequential citations
    assert assembled.documents[0].citation_id == 1
    assert assembled.documents[0].citation_label == "[1]"
    assert assembled.documents[1].citation_id == 2
    assert assembled.documents[1].citation_label == "[2]"

    # Verify citation map
    assert assembled.get_citation(1) == assembled.documents[0]
    assert assembled.get_citation(2) == assembled.documents[1]

    # Verify formatted context contains markers
    assert "[1]" in assembled.formatted_context
    assert "[2]" in assembled.formatted_context
    assert "Section 1 about PostgreSQL" in assembled.formatted_context


def test_context_builder_token_limit_strictly_enforced() -> None:
    # 5 large chunks
    chunks = [
        _make_chunk(
            f"This is a relatively long chunk number {i} " + "content repetition " * 40,
            score=0.9 - (i * 0.1),
        )
        for i in range(5)
    ]

    # Constrain to 150 tokens
    builder = ContextBuilder(max_tokens=150)
    assembled = builder.build_context(query="content", candidates=chunks)

    assert assembled.total_tokens <= 150
    assert assembled.total_chunks < len(chunks)
    assert assembled.truncated is True
    assert assembled.dropped_chunks_count > 0


def test_context_builder_chunk_limit_enforced() -> None:
    chunks = [_make_chunk(f"Short chunk {i}", score=0.9 - (i * 0.05)) for i in range(10)]

    builder = ContextBuilder(max_chunks=3, max_tokens=5000)
    assembled = builder.build_context(query="short", candidates=chunks)

    assert len(assembled.documents) == 3
    assert assembled.total_chunks == 3
    assert assembled.dropped_chunks_count == 7
    assert assembled.truncated is True


def test_context_builder_empty_candidates() -> None:
    builder = ContextBuilder()
    assembled = builder.build_context(query="anything", candidates=[])

    assert assembled.total_tokens == 0
    assert assembled.total_chunks == 0
    assert assembled.documents == []
    assert assembled.formatted_context == ""
    assert assembled.truncated is False


def test_context_builder_xml_formatting() -> None:
    c = _make_chunk("XML test chunk content.", score=0.85, page=4, doc_name="Manual.pdf")
    formatter = ContextFormatter(style="xml")
    builder = ContextBuilder(formatter=formatter)

    assembled = builder.build_context(query="test", candidates=[c])

    assert '<document citation="[1]"' in assembled.formatted_context
    assert 'title="Manual.pdf"' in assembled.formatted_context
    assert 'page="4"' in assembled.formatted_context
    assert "</document>" in assembled.formatted_context


def test_context_builder_page_number_sanitization() -> None:
    c1 = _make_chunk("Page int", page=5)
    c2 = _make_chunk("Page str", page="12")
    c3 = _make_chunk("Page invalid", page="invalid_page")
    c4 = _make_chunk("Page None", page=None)

    builder = ContextBuilder()
    assembled = builder.build_context(query="page", candidates=[c1, c2, c3, c4])

    docs_by_content = {d.content: d.page_number for d in assembled.documents}
    assert docs_by_content["Page int"] == 5
    assert docs_by_content["Page str"] == 12
    assert docs_by_content["Page invalid"] is None
    assert docs_by_content["Page None"] is None


def test_context_builder_determinism() -> None:
    chunks = [
        _make_chunk(f"Deterministic chunk {i} details.", score=0.5 + (i * 0.05)) for i in range(4)
    ]
    builder = ContextBuilder()

    res1 = builder.build_context(query="details", candidates=chunks)
    res2 = builder.build_context(query="details", candidates=chunks)

    assert res1.formatted_context == res2.formatted_context
    assert res1.total_tokens == res2.total_tokens
    assert [d.citation_id for d in res1.documents] == [d.citation_id for d in res2.documents]
