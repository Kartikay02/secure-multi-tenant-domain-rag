"""Unit tests for PGVectorStore vector storage, indexing, and similarity search."""

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.version import DocumentVersion
from app.rag.embeddings.vector_interfaces import VectorRecord
from app.rag.vector.pgvector import PGVectorStore


@pytest.mark.asyncio
async def test_pgvector_store_upsert_and_count(db_session: AsyncSession) -> None:
    """Verify batch vector upsert, get, and count operations."""
    # Setup document, version, and chunk in DB
    doc_id = uuid.uuid4()
    ver_id = uuid.uuid4()
    chunk_1_id = uuid.uuid4()
    chunk_2_id = uuid.uuid4()

    doc = Document(id=doc_id, name="Test Doc", document_type="md", source="test.md")
    ver = DocumentVersion(id=ver_id, document_id=doc_id, version_number=1, content_hash="hash1")
    c1 = DocumentChunk(
        id=chunk_1_id, document_version_id=ver_id, chunk_index=0, content="Content 1"
    )
    c2 = DocumentChunk(
        id=chunk_2_id, document_version_id=ver_id, chunk_index=1, content="Content 2"
    )

    db_session.add_all([doc, ver, c1, c2])
    await db_session.flush()

    vector_store = PGVectorStore(session=db_session, dimension=3)
    records = [
        VectorRecord(
            id=str(chunk_1_id),
            vector=[1.0, 0.0, 0.0],
            document_id=str(doc_id),
            metadata={"tag": "a"},
        ),
        VectorRecord(
            id=str(chunk_2_id),
            vector=[0.0, 1.0, 0.0],
            document_id=str(doc_id),
            metadata={"tag": "b"},
        ),
    ]

    upserted = await vector_store.upsert_vectors(records)
    assert upserted == 2
    assert await vector_store.count() == 2

    # Verify get_vector
    v1 = await vector_store.get_vector(chunk_1_id)
    assert v1 is not None
    assert v1.id == str(chunk_1_id)
    assert v1.vector == [1.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_pgvector_store_similarity_ranking_and_scores(db_session: AsyncSession) -> None:
    """Verify similarity search returns ranked results with normalized similarity scores."""
    doc_id = uuid.uuid4()
    ver_id = uuid.uuid4()
    c1_id = uuid.uuid4()
    c2_id = uuid.uuid4()

    doc = Document(id=doc_id, name="Guide", document_type="md", source="guide.md")
    ver = DocumentVersion(id=ver_id, document_id=doc_id, version_number=1, content_hash="hash2")
    c1 = DocumentChunk(
        id=c1_id, document_version_id=ver_id, chunk_index=0, content="PostgreSQL vector indexes"
    )
    c2 = DocumentChunk(
        id=c2_id, document_version_id=ver_id, chunk_index=1, content="Frontend React buttons"
    )

    db_session.add_all([doc, ver, c1, c2])
    await db_session.flush()

    vector_store = PGVectorStore(session=db_session, dimension=2)
    await vector_store.upsert_vectors(
        [
            VectorRecord(id=str(c1_id), vector=[1.0, 0.0], document_id=str(doc_id)),
            VectorRecord(id=str(c2_id), vector=[0.0, 1.0], document_id=str(doc_id)),
        ]
    )

    # Query vector aligned with c1 [1.0, 0.0]
    results = await vector_store.similarity_search(query_vector=[1.0, 0.0], top_k=2)
    assert len(results) == 2

    # Top result must be c1
    top = results[0]
    assert top.chunk_id == c1_id
    assert top.document_id == doc_id
    assert top.text == "PostgreSQL vector indexes"
    assert top.score == pytest.approx(1.0, rel=1e-2)
    assert 0.0 <= top.score <= 1.0
    assert "document_name" in top.metadata

    # Second result is c2 (orthogonal vector, score ~0.0)
    second = results[1]
    assert second.chunk_id == c2_id
    assert second.score == pytest.approx(0.0, abs=1e-2)


@pytest.mark.asyncio
async def test_pgvector_store_score_threshold_cutoff(db_session: AsyncSession) -> None:
    """Verify results below score_threshold are filtered out."""
    doc_id = uuid.uuid4()
    ver_id = uuid.uuid4()
    c1_id = uuid.uuid4()
    c2_id = uuid.uuid4()

    doc = Document(id=doc_id, name="Doc", document_type="txt", source="doc.txt")
    ver = DocumentVersion(id=ver_id, document_id=doc_id, version_number=1, content_hash="hash3")
    c1 = DocumentChunk(id=c1_id, document_version_id=ver_id, chunk_index=0, content="Match")
    c2 = DocumentChunk(id=c2_id, document_version_id=ver_id, chunk_index=1, content="Mismatch")

    db_session.add_all([doc, ver, c1, c2])
    await db_session.flush()

    vector_store = PGVectorStore(session=db_session, dimension=2)
    await vector_store.upsert_vectors(
        [
            VectorRecord(id=str(c1_id), vector=[1.0, 0.0], document_id=str(doc_id)),
            VectorRecord(id=str(c2_id), vector=[0.0, 1.0], document_id=str(doc_id)),
        ]
    )

    # With threshold 0.5, only c1 should survive
    results = await vector_store.similarity_search(
        query_vector=[1.0, 0.0],
        top_k=5,
        score_threshold=0.5,
    )
    assert len(results) == 1
    assert results[0].chunk_id == c1_id


@pytest.mark.asyncio
async def test_pgvector_store_metadata_filtering(db_session: AsyncSession) -> None:
    """Verify filtering by document_id and custom metadata tags."""
    doc1_id = uuid.uuid4()
    doc2_id = uuid.uuid4()
    ver1_id = uuid.uuid4()
    ver2_id = uuid.uuid4()
    c1_id = uuid.uuid4()
    c2_id = uuid.uuid4()

    d1 = Document(id=doc1_id, name="Doc 1", document_type="md", source="d1.md")
    d2 = Document(id=doc2_id, name="Doc 2", document_type="md", source="d2.md")
    v1 = DocumentVersion(id=ver1_id, document_id=doc1_id, version_number=1, content_hash="h1")
    v2 = DocumentVersion(id=ver2_id, document_id=doc2_id, version_number=1, content_hash="h2")
    c1 = DocumentChunk(
        id=c1_id,
        document_version_id=ver1_id,
        chunk_index=0,
        content="Sec 1",
        metadata_json={"category": "finance"},
    )
    c2 = DocumentChunk(
        id=c2_id,
        document_version_id=ver2_id,
        chunk_index=0,
        content="Sec 2",
        metadata_json={"category": "legal"},
    )

    db_session.add_all([d1, d2, v1, v2, c1, c2])
    await db_session.flush()

    vector_store = PGVectorStore(session=db_session, dimension=2)
    await vector_store.upsert_vectors(
        [
            VectorRecord(
                id=str(c1_id),
                vector=[1.0, 0.0],
                document_id=str(doc1_id),
                metadata={"category": "finance"},
            ),
            VectorRecord(
                id=str(c2_id),
                vector=[1.0, 0.0],
                document_id=str(doc2_id),
                metadata={"category": "legal"},
            ),
        ]
    )

    # 1. Filter by document_id
    res_doc = await vector_store.similarity_search(
        query_vector=[1.0, 0.0],
        filter_metadata={"document_id": str(doc1_id)},
    )
    assert len(res_doc) == 1
    assert res_doc[0].chunk_id == c1_id

    # 2. Filter by category
    res_cat = await vector_store.similarity_search(
        query_vector=[1.0, 0.0],
        filter_metadata={"category": "legal"},
    )
    assert len(res_cat) == 1
    assert res_cat[0].chunk_id == c2_id


@pytest.mark.asyncio
async def test_pgvector_store_deletion(db_session: AsyncSession) -> None:
    """Verify vector deletion by chunk IDs and by document ID."""
    doc_id = uuid.uuid4()
    ver_id = uuid.uuid4()
    c1_id = uuid.uuid4()
    c2_id = uuid.uuid4()

    doc = Document(id=doc_id, name="Doc", document_type="txt", source="doc.txt")
    ver = DocumentVersion(id=ver_id, document_id=doc_id, version_number=1, content_hash="hdel")
    c1 = DocumentChunk(id=c1_id, document_version_id=ver_id, chunk_index=0, content="C1")
    c2 = DocumentChunk(id=c2_id, document_version_id=ver_id, chunk_index=1, content="C2")

    db_session.add_all([doc, ver, c1, c2])
    await db_session.flush()

    vector_store = PGVectorStore(session=db_session, dimension=2)
    await vector_store.upsert_vectors(
        [
            VectorRecord(id=str(c1_id), vector=[1.0, 0.0], document_id=str(doc_id)),
            VectorRecord(id=str(c2_id), vector=[0.0, 1.0], document_id=str(doc_id)),
        ]
    )
    assert await vector_store.count() == 2

    # Delete single vector by chunk_id
    deleted = await vector_store.delete_vectors([c1_id])
    assert deleted == 1
    assert await vector_store.count() == 1
    assert await vector_store.get_vector(c1_id) is None

    # Delete all by document
    deleted_doc = await vector_store.delete_by_document(doc_id)
    assert deleted_doc == 1
    assert await vector_store.count() == 0


@pytest.mark.asyncio
async def test_vector_store_factory_mock_provider() -> None:
    """Verify VectorStoreFactory instantiates InMemoryVectorStore when provider is mock (SEC-43)."""
    from app.core.config import Settings
    from app.rag.vector.factory import VectorStoreFactory
    from app.rag.vector.in_memory import InMemoryVectorStore

    VectorStoreFactory.reset_mock()
    settings = Settings()
    settings.vector_store.provider = "mock"

    store = VectorStoreFactory.create(settings=settings)
    assert isinstance(store, InMemoryVectorStore)


@pytest.mark.asyncio
async def test_vector_store_factory_rejects_pgvector_on_sqlite() -> None:
    """Verify VectorStoreFactory raises ConfigurationError for pgvector on SQLite (SEC-13, SEC-43)."""
    from app.core.config import Settings
    from app.core.exceptions import ConfigurationError
    from app.rag.vector.factory import VectorStoreFactory

    settings = Settings()
    settings.database.url = "sqlite+aiosqlite:///./test.db"
    settings.vector_store.provider = "pgvector"

    with pytest.raises(ConfigurationError) as exc_info:
        VectorStoreFactory.create(settings=settings)
    assert "incompatible with sqlite" in exc_info.value.message.lower()


@pytest.mark.asyncio
async def test_vector_store_factory_rejects_unknown_provider() -> None:
    """Verify VectorStoreFactory raises ConfigurationError for unsupported provider (SEC-43)."""
    from app.core.config import Settings
    from app.core.exceptions import ConfigurationError
    from app.rag.vector.factory import VectorStoreFactory

    settings = Settings()
    settings.vector_store.provider = "unsupported-provider"  # type: ignore[assignment]

    with pytest.raises(ConfigurationError) as exc_info:
        VectorStoreFactory.create(settings=settings)
    assert "unsupported vector_store_provider" in exc_info.value.message.lower()


@pytest.mark.asyncio
async def test_in_memory_vector_store_crud() -> None:
    """Verify InMemoryVectorStore upsert, get, delete, and similarity search operations."""
    from app.rag.embeddings.vector_interfaces import VectorRecord
    from app.rag.vector.in_memory import InMemoryVectorStore

    store = InMemoryVectorStore(dimension=3)
    chunk_1 = uuid.uuid4()
    chunk_2 = uuid.uuid4()
    doc_id = uuid.uuid4()

    records = [
        VectorRecord(
            id=str(chunk_1),
            vector=[1.0, 0.0, 0.0],
            document_id=str(doc_id),
            metadata={"tag": "first"},
        ),
        VectorRecord(
            id=str(chunk_2),
            vector=[0.0, 1.0, 0.0],
            document_id=str(doc_id),
            metadata={"tag": "second"},
        ),
    ]

    upserted = await store.upsert_vectors(records)
    assert upserted == 2
    assert await store.count() == 2

    # Get
    v1 = await store.get_vector(chunk_1)
    assert v1 is not None
    assert v1.id == str(chunk_1)

    # Search
    results = await store.similarity_search(query_vector=[0.9, 0.1, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == chunk_1

    # Delete
    deleted = await store.delete_vectors([chunk_1])
    assert deleted == 1
    assert await store.count() == 1

    # Delete by document
    del_doc = await store.delete_by_document(doc_id)
    assert del_doc == 1
    assert await store.count() == 0
