"""Unit tests for MockEmbeddingProvider and InMemoryVectorStore."""

import math

import pytest

from app.core.exceptions import EmbeddingError
from app.rag.embeddings.mock import MockEmbeddingProvider
from app.rag.embeddings.vector_interfaces import VectorRecord
from app.rag.embeddings.vector_mock import InMemoryVectorStore


@pytest.mark.asyncio
async def test_mock_embedding_determinism() -> None:
    """Verify identical text produces identical vectors, and different texts produce distinct vectors."""
    provider = MockEmbeddingProvider(dimension=128)

    v1 = await provider.embed_text("Clean Architecture in Python")
    v2 = await provider.embed_text("Clean Architecture in Python")
    v3 = await provider.embed_text("Something completely different")

    assert v1 == v2
    assert v1 != v3
    assert len(v1) == 128


@pytest.mark.asyncio
async def test_mock_embedding_l2_normalization() -> None:
    """Verify output vectors are unit normalized (L2 norm = 1.0)."""
    provider = MockEmbeddingProvider(dimension=256)
    vector = await provider.embed_text("Vector normalization test")

    norm = math.sqrt(sum(x * x for x in vector))
    assert pytest.approx(norm, rel=1e-3) == 1.0


@pytest.mark.asyncio
async def test_mock_embedding_dimensions() -> None:
    """Verify provider respects custom dimensions."""
    for dim in [64, 384, 768, 1536]:
        provider = MockEmbeddingProvider(dimension=dim)
        assert provider.dimension == dim
        vec = await provider.embed_text("Test query")
        assert len(vec) == dim


@pytest.mark.asyncio
async def test_mock_embedding_batching() -> None:
    """Verify batching across document lists larger than batch_size."""
    provider = MockEmbeddingProvider(dimension=64, batch_size=10)
    texts = [f"Document chunk number {i}" for i in range(35)]

    vectors = await provider.embed_documents(texts)
    assert len(vectors) == 35
    for vec in vectors:
        assert len(vec) == 64
        norm = math.sqrt(sum(x * x for x in vec))
        assert pytest.approx(norm, rel=1e-3) == 1.0

    # Provider should have executed 4 batch calls for 35 items with batch_size 10
    assert provider.total_calls == 4
    assert provider.embedded_text_count == 35


@pytest.mark.asyncio
async def test_mock_embedding_empty_input() -> None:
    """Verify handling of empty inputs."""
    provider = MockEmbeddingProvider(dimension=64)
    assert await provider.embed_documents([]) == []

    # Empty string still produces a valid deterministic vector
    vec = await provider.embed_text("")
    assert len(vec) == 64


@pytest.mark.asyncio
async def test_mock_embedding_simulated_failure() -> None:
    """Verify simulated persistent failure raises EmbeddingError."""
    provider = MockEmbeddingProvider(simulate_failure=True)
    with pytest.raises(EmbeddingError, match="Simulated persistent"):
        await provider.embed_text("Will fail")


@pytest.mark.asyncio
async def test_mock_embedding_transient_failure_recovery() -> None:
    """Verify simulated transient failure count fails N times then recovers."""
    provider = MockEmbeddingProvider(failure_count=2)

    # Call 1: fails
    with pytest.raises(EmbeddingError, match="Simulated transient"):
        await provider.embed_text("Attempt 1")

    # Call 2: fails
    with pytest.raises(EmbeddingError, match="Simulated transient"):
        await provider.embed_text("Attempt 2")

    # Call 3: succeeds
    vec = await provider.embed_text("Attempt 3")
    assert len(vec) == 384


@pytest.mark.asyncio
async def test_in_memory_vector_store() -> None:
    """Verify InMemoryVectorStore upsert, get, delete, count, and clear operations."""
    store = InMemoryVectorStore()
    assert await store.count() == 0

    records = [
        VectorRecord(id="r1", vector=[0.1, 0.2], document_id="d1", version_id="v1"),
        VectorRecord(id="r2", vector=[0.3, 0.4], document_id="d1", version_id="v1"),
    ]

    upserted = await store.upsert(records)
    assert upserted == 2
    assert await store.count() == 2

    # Get
    r1 = await store.get("r1")
    assert r1 is not None
    assert r1.id == "r1"
    assert r1.vector == [0.1, 0.2]

    # Delete
    deleted = await store.delete(["r1"])
    assert deleted == 1
    assert await store.count() == 1
    assert await store.get("r1") is None

    # Clear
    store.clear()
    assert await store.count() == 0
