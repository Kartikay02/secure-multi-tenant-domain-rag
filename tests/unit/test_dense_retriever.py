"""Unit tests for DenseRetriever."""

import uuid
from typing import Any

import pytest

from app.rag.embeddings.mock import MockEmbeddingProvider
from app.rag.embeddings.vector_interfaces import VectorRecord
from app.rag.retrieval.dense import DenseRetriever
from app.rag.vector.domain import RetrievalResult
from app.rag.vector.interfaces import VectorStoreProtocol


class MockVectorStore(VectorStoreProtocol):
    async def upsert_vectors(self, records: list[VectorRecord]) -> int:
        return len(records)

    async def similarity_search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk_id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                text="Vector match",
                score=0.92,
            )
        ]

    async def delete_vectors(self, chunk_ids: list[uuid.UUID]) -> int:
        return len(chunk_ids)

    async def delete_by_document(self, document_id: uuid.UUID) -> int:
        return 1

    async def get_vector(self, chunk_id: uuid.UUID) -> VectorRecord | None:
        return None

    async def count(self) -> int:
        return 1


@pytest.mark.asyncio
async def test_dense_retriever_delegation_and_scoring() -> None:
    """Verify DenseRetriever generates query embedding, queries vector store, and stamps dense_score."""
    embedding_provider = MockEmbeddingProvider(dimension=64)
    vector_store = MockVectorStore()
    retriever = DenseRetriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    results = await retriever.retrieve("Machine learning search", top_k=3)
    assert len(results) == 1
    assert results[0].text == "Vector match"
    assert results[0].score == 0.92
    assert results[0].dense_score == 0.92
    assert results[0].sparse_score is None


@pytest.mark.asyncio
async def test_dense_retriever_empty_query() -> None:
    """Verify empty query returns empty list without calling provider."""
    embedding_provider = MockEmbeddingProvider(dimension=64)
    vector_store = MockVectorStore()
    retriever = DenseRetriever(embedding_provider=embedding_provider, vector_store=vector_store)

    assert await retriever.retrieve("") == []
    assert await retriever.retrieve("   ") == []
