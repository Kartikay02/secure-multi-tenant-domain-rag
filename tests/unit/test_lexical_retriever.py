"""Unit tests for PGLexicalRetriever."""

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.version import DocumentVersion
from app.rag.retrieval.lexical import PGLexicalRetriever


@pytest.mark.asyncio
async def test_lexical_retriever_exact_and_partial_match(db_session: AsyncSession) -> None:
    """Verify lexical retrieval matches keywords and ranks by term density."""
    doc_id = uuid.uuid4()
    ver_id = uuid.uuid4()
    c1_id = uuid.uuid4()
    c2_id = uuid.uuid4()
    c3_id = uuid.uuid4()

    doc = Document(id=doc_id, name="Doc", document_type="txt", source="doc.txt")
    ver = DocumentVersion(id=ver_id, document_id=doc_id, version_number=1, content_hash="hlex")
    c1 = DocumentChunk(
        id=c1_id,
        document_version_id=ver_id,
        chunk_index=0,
        content="PostgreSQL database WAL replication and archiving.",
    )
    c2 = DocumentChunk(
        id=c2_id,
        document_version_id=ver_id,
        chunk_index=1,
        content="PostgreSQL database query optimization.",
    )
    c3 = DocumentChunk(
        id=c3_id,
        document_version_id=ver_id,
        chunk_index=2,
        content="Frontend JavaScript styling with Tailwind CSS.",
    )

    db_session.add_all([doc, ver, c1, c2, c3])
    await db_session.flush()

    retriever = PGLexicalRetriever(session=db_session)

    # Search for "WAL replication"
    results = await retriever.retrieve("WAL replication", top_k=5)
    assert len(results) == 1
    assert results[0].chunk_id == c1_id
    assert results[0].sparse_score is not None
    assert results[0].sparse_score > 0.0
    assert results[0].dense_score is None

    # Search for "PostgreSQL database"
    res_both = await retriever.retrieve("PostgreSQL database", top_k=5)
    assert len(res_both) == 2
    chunk_ids = {r.chunk_id for r in res_both}
    assert chunk_ids == {c1_id, c2_id}


@pytest.mark.asyncio
async def test_lexical_retriever_metadata_filtering(db_session: AsyncSession) -> None:
    """Verify lexical retriever respects metadata filters."""
    d1_id = uuid.uuid4()
    d2_id = uuid.uuid4()
    v1_id = uuid.uuid4()
    v2_id = uuid.uuid4()
    c1_id = uuid.uuid4()
    c2_id = uuid.uuid4()

    d1 = Document(id=d1_id, name="D1", document_type="md", source="d1.md")
    d2 = Document(id=d2_id, name="D2", document_type="md", source="d2.md")
    v1 = DocumentVersion(id=v1_id, document_id=d1_id, version_number=1, content_hash="v1")
    v2 = DocumentVersion(id=v2_id, document_id=d2_id, version_number=1, content_hash="v2")
    c1 = DocumentChunk(
        id=c1_id,
        document_version_id=v1_id,
        chunk_index=0,
        content="Compliance policy manual.",
        metadata_json={"department": "HR"},
    )
    c2 = DocumentChunk(
        id=c2_id,
        document_version_id=v2_id,
        chunk_index=0,
        content="Compliance policy manual.",
        metadata_json={"department": "Finance"},
    )

    db_session.add_all([d1, d2, v1, v2, c1, c2])
    await db_session.flush()

    retriever = PGLexicalRetriever(session=db_session)

    # Filter by department="HR"
    res_hr = await retriever.retrieve(
        query="Compliance policy",
        filter_metadata={"department": "HR"},
    )
    assert len(res_hr) == 1
    assert res_hr[0].chunk_id == c1_id


@pytest.mark.asyncio
async def test_lexical_retriever_empty_query(db_session: AsyncSession) -> None:
    """Verify empty or whitespace queries return empty list."""
    retriever = PGLexicalRetriever(session=db_session)
    assert await retriever.retrieve("") == []
    assert await retriever.retrieve("   \t  ") == []
