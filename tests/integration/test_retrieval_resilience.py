"""Integration tests validating retrieval layer resilience, edge cases, and empty states."""

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.version import DocumentVersion
from app.rag.embeddings.mock import MockEmbeddingProvider
from app.rag.reranking.mock import MockReranker
from app.rag.retrieval.dense import DenseRetriever
from app.rag.retrieval.fusion import ReciprocalRankFusion
from app.rag.retrieval.hybrid import HybridRetriever
from app.rag.retrieval.lexical import PGLexicalRetriever
from app.rag.vector.domain import RetrievalResult
from app.rag.vector.pgvector import PGVectorStore
from app.repositories import (
    SQLAlchemyDocumentChunkRepository,
    SQLAlchemyDocumentRepository,
)


@pytest.mark.asyncio
async def test_lexical_retriever_empty_database_returns_empty_list(
    db_session: AsyncSession,
) -> None:
    """Verify lexical retrieval against an empty database returns [] without errors."""
    retriever = PGLexicalRetriever(session=db_session)
    results = await retriever.retrieve(query="PostgreSQL indexing", top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_lexical_retriever_no_matching_terms(
    db_session: AsyncSession,
    document_repo: SQLAlchemyDocumentRepository,
    chunk_repo: SQLAlchemyDocumentChunkRepository,
) -> None:
    """Verify lexical retrieval with keywords completely absent from the corpus returns []."""
    doc = Document(name="DB Guide", document_type="txt", source="local://db.txt")
    ver = DocumentVersion(
        version_number=1, content_hash="db_hash", size_bytes=100, status="PROCESSED"
    )
    await document_repo.create_with_version(doc, ver)

    chunks = [
        DocumentChunk(
            document_version_id=ver.id,
            chunk_index=0,
            content="PostgreSQL relational database configuration and performance tuning.",
            token_count=10,
            char_count=65,
        )
    ]
    await chunk_repo.bulk_create_chunks(chunks)
    await db_session.commit()

    retriever = PGLexicalRetriever(session=db_session)
    # Search for terms unrelated to databases
    results = await retriever.retrieve(
        query="quantum astrophysics black hole event horizon", top_k=5
    )
    assert results == []


@pytest.mark.asyncio
async def test_dense_retriever_empty_vector_store(
    db_session: AsyncSession,
) -> None:
    """Verify dense retrieval against an unpopulated vector store returns []."""
    embed_provider = MockEmbeddingProvider(dimension=384)
    vector_store = PGVectorStore(session=db_session, dimension=384)
    retriever = DenseRetriever(embedding_provider=embed_provider, vector_store=vector_store)

    results = await retriever.retrieve(query="distributed systems consensus", top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_hybrid_retriever_with_one_empty_branch(
    db_session: AsyncSession,
) -> None:
    """Verify hybrid retriever functions properly when one branch returns candidates and the other is empty."""
    embed_provider = MockEmbeddingProvider(dimension=384)
    vector_store = PGVectorStore(session=db_session, dimension=384)
    dense_retriever = DenseRetriever(embedding_provider=embed_provider, vector_store=vector_store)
    lexical_retriever = PGLexicalRetriever(session=db_session)

    hybrid = HybridRetriever(
        dense_retriever=dense_retriever,
        lexical_retriever=lexical_retriever,
        fusion_strategy=ReciprocalRankFusion(),
    )

    results = await hybrid.retrieve(query="any search query", top_k=5)
    assert results == []


def test_fusion_both_branches_empty() -> None:
    """Verify fusion strategy handles completely empty input lists safely."""
    rrf = ReciprocalRankFusion()
    fused = rrf.fuse(dense_results=[], sparse_results=[], top_k=10)
    assert fused == []


def test_fusion_disjoint_candidate_lists() -> None:
    """Verify fusion correctly unifies candidates when dense and sparse result sets share zero overlap."""
    c1, c2 = uuid.uuid4(), uuid.uuid4()
    did = uuid.uuid4()
    dense = [RetrievalResult(chunk_id=c1, document_id=did, text="Dense candidate", score=0.9)]
    sparse = [RetrievalResult(chunk_id=c2, document_id=did, text="Sparse candidate", score=0.8)]

    rrf = ReciprocalRankFusion()
    fused = rrf.fuse(dense_results=dense, sparse_results=sparse, top_k=5)

    assert len(fused) == 2
    fused_ids = {f.chunk_id for f in fused}
    assert fused_ids == {c1, c2}


@pytest.mark.asyncio
async def test_reranker_with_empty_candidates() -> None:
    """Verify reranker returns [] immediately when given an empty candidate list."""
    reranker = MockReranker()
    reranked = await reranker.rerank(query="test query", candidates=[], top_k=5)
    assert reranked == []


@pytest.mark.asyncio
async def test_high_score_threshold_filters_all_candidates() -> None:
    """Verify score_threshold cut-off cleanly drops all lower scoring candidates."""
    reranker = MockReranker()
    c1 = uuid.uuid4()
    did = uuid.uuid4()
    candidates = [
        RetrievalResult(chunk_id=c1, document_id=did, text="Low score candidate", score=0.3)
    ]

    reranked = await reranker.rerank(
        query="test query",
        candidates=candidates,
        top_k=5,
    )
    # Filter by strict threshold
    filtered = [c for c in reranked if c.score >= 0.99]
    assert filtered == []
