"""PostgreSQL-backed vector store implementation using pgvector and SQLAlchemy 2.0."""

import math
import uuid
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import DatabaseError
from app.core.logging import get_logger
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.embedding import ChunkEmbedding
from app.rag.embeddings.vector_interfaces import VectorRecord
from app.rag.vector.domain import RetrievalResult
from app.rag.vector.interfaces import VectorStoreProtocol

logger = get_logger("app.rag.vector.pgvector")


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two float vectors in Python."""
    if len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2, strict=True))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class PGVectorStore(VectorStoreProtocol):
    """PostgreSQL pgvector storage adapter implementing VectorStoreProtocol.

    Features HNSW index acceleration, GIN metadata indexing, batch upserts,
    cascading document/chunk lifecycle management, and transparent test fallback.
    """

    def __init__(
        self,
        session: AsyncSession | None = None,
        model_name: str = "default",
        dimension: int = 384,
    ) -> None:
        self._explicit_session = session
        self.model_name = model_name
        self.dimension = dimension

    @property
    def session(self) -> AsyncSession:
        if self._explicit_session is not None:
            return self._explicit_session
        from app.core.database import get_current_session

        current = get_current_session()
        if current is not None:
            return current
        raise DatabaseError("No active database session available for PGVectorStore.")

    @session.setter
    def session(self, value: AsyncSession | None) -> None:
        self._explicit_session = value

    def _is_postgresql(self) -> bool:
        """Check if active connection dialect is PostgreSQL."""
        bind = self.session.bind
        if bind and hasattr(bind, "dialect"):
            return str(bind.dialect.name).lower() == "postgresql"
        return False

    async def upsert(self, records: list[VectorRecord]) -> int:
        """Alias conforming to embeddings VectorStoreProtocol."""
        return await self.upsert_vectors(records)

    async def get(self, record_id: str) -> VectorRecord | None:
        """Alias conforming to embeddings VectorStoreProtocol."""
        try:
            return await self.get_vector(uuid.UUID(record_id))
        except (ValueError, TypeError):
            return None

    async def delete(self, record_ids: list[str]) -> int:
        """Alias conforming to embeddings VectorStoreProtocol."""
        chunk_uuids = [uuid.UUID(rid) for rid in record_ids if rid]
        return await self.delete_vectors(chunk_uuids)

    async def upsert_vectors(self, records: list[VectorRecord]) -> int:
        """Insert or update batch of vector records into chunk_embeddings.

        Args:
            records: List of VectorRecord instances to persist.

        Returns:
            Count of successfully upserted records.
        """
        if not records:
            return 0

        try:
            # Fetch existing records to decide insert vs update
            chunk_ids = [uuid.UUID(r.id) for r in records]
            query = select(ChunkEmbedding).where(ChunkEmbedding.chunk_id.in_(chunk_ids))
            result = await self.session.execute(query)
            existing_map = {e.chunk_id: e for e in result.scalars().all()}

            count = 0
            for r in records:
                c_id = uuid.UUID(r.id)
                d_id = uuid.UUID(r.document_id) if r.document_id else None

                # Fallback to look up document_id from chunk if omitted
                if d_id is None:
                    chunk_row = await self.session.get(DocumentChunk, c_id)
                    if chunk_row and chunk_row.version:
                        d_id = chunk_row.version.document_id

                if d_id is None:
                    logger.warning(f"Could not resolve document_id for chunk {c_id}, skipping.")
                    continue

                if c_id in existing_map:
                    # Update existing embedding entry
                    entry = existing_map[c_id]
                    entry.embedding = r.vector
                    entry.model = self.model_name
                    entry.dimension = len(r.vector)
                    entry.metadata_json = r.metadata
                else:
                    # Insert new embedding entry
                    new_entry = ChunkEmbedding(
                        id=uuid.uuid4(),
                        chunk_id=c_id,
                        document_id=d_id,
                        embedding=r.vector,
                        model=self.model_name,
                        dimension=len(r.vector),
                        metadata_json=r.metadata,
                    )
                    self.session.add(new_entry)
                count += 1

            await self.session.flush()
            logger.debug(f"Upserted {count} vector records into chunk_embeddings.")
            return count

        except SQLAlchemyError as exc:
            await self.session.rollback()
            logger.error(f"Error upserting vectors: {exc}")
            raise DatabaseError(f"Database error upserting vectors: {exc}") from exc

    async def similarity_search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Perform nearest-neighbor vector search with metadata filtering.

        Args:
            query_vector: Query embedding vector.
            top_k: Max candidate chunks to return.
            score_threshold: Minimum similarity threshold [0.0, 1.0].
            filter_metadata: Constraints dictionary (e.g. {"document_id": "...", "section": "..."}).

        Returns:
            List of RetrievalResult objects sorted descending by score.
        """
        if not query_vector:
            return []

        try:
            is_pg = self._is_postgresql()

            if is_pg:
                # 1. Native PostgreSQL pgvector search
                distance_expr = ChunkEmbedding.embedding.cosine_distance(query_vector)
                stmt = (
                    select(
                        ChunkEmbedding,
                        DocumentChunk,
                        Document,
                        distance_expr.label("distance"),
                    )
                    .join(DocumentChunk, ChunkEmbedding.chunk_id == DocumentChunk.id)
                    .join(Document, ChunkEmbedding.document_id == Document.id)
                )

                # Metadata and document ID filtering
                if filter_metadata:
                    if "document_id" in filter_metadata:
                        doc_uuid = uuid.UUID(str(filter_metadata["document_id"]))
                        stmt = stmt.where(ChunkEmbedding.document_id == doc_uuid)
                    for k, v in filter_metadata.items():
                        if k != "document_id":
                            stmt = stmt.where(ChunkEmbedding.metadata_json.contains({k: v}))

                stmt = stmt.order_by(distance_expr.asc()).limit(top_k * 2)
                result = await self.session.execute(stmt)
                rows = result.all()

                retrieval_results: list[RetrievalResult] = []
                for emb, chunk, doc, dist in rows:
                    raw_dist = float(dist) if dist is not None else 1.0
                    sim_score = max(0.0, min(1.0, 1.0 - raw_dist))

                    if score_threshold is not None and sim_score < score_threshold:
                        continue

                    combined_meta = dict(chunk.metadata_json or {})
                    combined_meta.update(emb.metadata_json or {})
                    combined_meta["document_name"] = doc.name
                    combined_meta["document_source"] = doc.source
                    combined_meta["chunk_index"] = chunk.chunk_index

                    retrieval_results.append(
                        RetrievalResult(
                            chunk_id=chunk.id,
                            document_id=doc.id,
                            text=chunk.content,
                            score=round(sim_score, 4),
                            metadata=combined_meta,
                        )
                    )

                retrieval_results.sort(key=lambda r: r.score, reverse=True)
                return retrieval_results[:top_k]

            else:
                # 2. SQLite / in-memory fallback for local testing
                stmt = (
                    select(ChunkEmbedding, DocumentChunk, Document)
                    .join(DocumentChunk, ChunkEmbedding.chunk_id == DocumentChunk.id)
                    .join(Document, ChunkEmbedding.document_id == Document.id)
                )

                if filter_metadata:
                    if "document_id" in filter_metadata:
                        doc_uuid = uuid.UUID(str(filter_metadata["document_id"]))
                        stmt = stmt.where(ChunkEmbedding.document_id == doc_uuid)

                result = await self.session.execute(stmt)
                rows = result.all()

                candidates: list[RetrievalResult] = []
                for emb, chunk, doc in rows:
                    # Apply arbitrary metadata filters
                    if filter_metadata:
                        skip = False
                        for k, v in filter_metadata.items():
                            if k == "document_id":
                                continue
                            if emb.metadata_json.get(k) != v and chunk.metadata_json.get(k) != v:
                                skip = True
                                break
                        if skip:
                            continue

                    emb_vec = list(emb.embedding)
                    cos_sim = _cosine_similarity(query_vector, emb_vec)
                    sim_score = max(0.0, min(1.0, cos_sim))

                    if score_threshold is not None and sim_score < score_threshold:
                        continue

                    combined_meta = dict(chunk.metadata_json or {})
                    combined_meta.update(emb.metadata_json or {})
                    combined_meta["document_name"] = doc.name
                    combined_meta["document_source"] = doc.source
                    combined_meta["chunk_index"] = chunk.chunk_index

                    candidates.append(
                        RetrievalResult(
                            chunk_id=chunk.id,
                            document_id=doc.id,
                            text=chunk.content,
                            score=round(sim_score, 4),
                            metadata=combined_meta,
                        )
                    )

                candidates.sort(key=lambda r: r.score, reverse=True)
                return candidates[:top_k]

        except SQLAlchemyError as exc:
            logger.error(f"Error during similarity search: {exc}")
            raise DatabaseError(f"Database error during similarity search: {exc}") from exc

    async def delete_vectors(self, chunk_ids: list[uuid.UUID]) -> int:
        """Delete vector records by chunk IDs."""
        if not chunk_ids:
            return 0
        try:
            stmt = delete(ChunkEmbedding).where(ChunkEmbedding.chunk_id.in_(chunk_ids))
            result = await self.session.execute(stmt)
            await self.session.flush()
            rowcount = getattr(result, "rowcount", 0)
            return int(rowcount if rowcount is not None else 0)
        except SQLAlchemyError as exc:
            await self.session.rollback()
            logger.error(f"Error deleting vectors: {exc}")
            raise DatabaseError(f"Database error deleting vectors: {exc}") from exc

    async def delete_by_document(self, document_id: uuid.UUID) -> int:
        """Delete all vector embeddings belonging to a document."""
        try:
            stmt = delete(ChunkEmbedding).where(ChunkEmbedding.document_id == document_id)
            result = await self.session.execute(stmt)
            await self.session.flush()
            rowcount = getattr(result, "rowcount", 0)
            return int(rowcount if rowcount is not None else 0)
        except SQLAlchemyError as exc:
            await self.session.rollback()
            logger.error(f"Error deleting vectors by document {document_id}: {exc}")
            raise DatabaseError(f"Database error deleting vectors by document: {exc}") from exc

    async def get_vector(self, chunk_id: uuid.UUID) -> VectorRecord | None:
        """Fetch a stored vector record by chunk ID."""
        try:
            query = select(ChunkEmbedding).where(ChunkEmbedding.chunk_id == chunk_id)
            result = await self.session.execute(query)
            entry = result.scalar_one_or_none()
            if not entry:
                return None
            return VectorRecord(
                id=str(entry.chunk_id),
                vector=list(entry.embedding),
                document_id=str(entry.document_id),
                metadata=entry.metadata_json,
            )
        except SQLAlchemyError as exc:
            logger.error(f"Error fetching vector for chunk {chunk_id}: {exc}")
            raise DatabaseError(f"Database error fetching vector: {exc}") from exc

    async def count(self) -> int:
        """Count total vectors indexed in chunk_embeddings."""
        try:
            query = select(func.count(ChunkEmbedding.id))
            result = await self.session.execute(query)
            return int(result.scalar_one() or 0)
        except SQLAlchemyError as exc:
            logger.error(f"Error counting vectors: {exc}")
            raise DatabaseError(f"Database error counting vectors: {exc}") from exc
