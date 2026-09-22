"""Unit tests for ContextDocument and AssembledContext domain models."""

import uuid

from app.rag.context.domain import AssembledContext, ContextDocument


def _make_doc(
    citation_id: int,
    title: str = "Doc",
    content: str = "Sample content",
    page: int | None = 1,
    score: float = 0.9,
) -> ContextDocument:
    return ContextDocument(
        citation_id=citation_id,
        citation_label=f"[{citation_id}]",
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        source_id="manual.pdf",
        title=title,
        page_number=page,
        score=score,
        content=content,
        token_count=10,
        dense_score=score,
        sparse_score=score,
        rerank_score=score,
    )


def test_context_document_creation() -> None:
    doc = _make_doc(citation_id=1, title="Architecture Overview", page=3, score=0.95)
    assert doc.citation_id == 1
    assert doc.citation_label == "[1]"
    assert doc.title == "Architecture Overview"
    assert doc.page_number == 3
    assert doc.score == 0.95
    assert doc.source_id == "manual.pdf"
    assert doc.dense_score == 0.95
    assert doc.sparse_score == 0.95
    assert doc.rerank_score == 0.95


def test_assembled_context_citation_map_and_lookup() -> None:
    doc1 = _make_doc(1, title="Section 1", score=0.9)
    doc2 = _make_doc(2, title="Section 2", score=0.8)

    assembled = AssembledContext(
        formatted_context="[1] Content 1\n\n[2] Content 2",
        documents=[doc1, doc2],
        citation_map={1: doc1, 2: doc2},
        total_tokens=20,
        total_chunks=2,
        truncated=False,
        dropped_chunks_count=0,
    )

    assert assembled.get_citation(1) == doc1
    assert assembled.get_citation(2) == doc2
    assert assembled.get_citation(3) is None


def test_assembled_context_citation_manifest() -> None:
    doc1 = _make_doc(1, title="WAL Replication", page=5, score=0.925)
    doc2 = _make_doc(2, title="Raft Consensus", page=None, score=0.812)

    assembled = AssembledContext(
        formatted_context="...",
        documents=[doc1, doc2],
        citation_map={1: doc1, 2: doc2},
        total_tokens=25,
        total_chunks=2,
        truncated=False,
        dropped_chunks_count=0,
    )

    manifest = assembled.format_citation_manifest()
    assert '[1] "WAL Replication", Page 5 (Source: manual.pdf, Score: 0.925)' in manifest
    assert '[2] "Raft Consensus" (Source: manual.pdf, Score: 0.812)' in manifest


def test_assembled_context_resolve_citations() -> None:
    doc1 = _make_doc(1, title="Section 1")
    doc2 = _make_doc(2, title="Section 2")
    doc3 = _make_doc(3, title="Section 3")

    assembled = AssembledContext(
        formatted_context="...",
        documents=[doc1, doc2, doc3],
        citation_map={1: doc1, 2: doc2, 3: doc3},
        total_tokens=30,
        total_chunks=3,
        truncated=False,
        dropped_chunks_count=0,
    )

    llm_output = (
        "PostgreSQL achieves durability via WAL [1]. "
        "High availability is maintained via streaming replication [1] and Raft consensus [3]. "
        "Non-existent citation [99] should be ignored."
    )

    resolved = assembled.resolve_citations(llm_output)
    # Should resolve [1] and [3] deduplicated in order of appearance
    assert len(resolved) == 2
    assert resolved[0] == doc1
    assert resolved[1] == doc3
