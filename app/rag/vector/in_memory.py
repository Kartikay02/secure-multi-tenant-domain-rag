"""In-memory vector store implementation for testing, dev, and SQLite mock modes."""

import math
import uuid
from typing import Any

from app.rag.embeddings.vector_interfaces import VectorRecord
from app.rag.vector.domain import RetrievalResult
from app.rag.vector.interfaces import VectorStoreProtocol


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2, strict=True))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorStore(VectorStoreProtocol):
    """In-memory vector store implementing VectorStoreProtocol."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self._records: dict[str, VectorRecord] = {}

    async def upsert_vectors(self, records: list[VectorRecord]) -> int:
        """Insert or update vector records."""
        for r in records:
            self._records[str(r.id)] = r
        return len(records)

    async def upsert(self, records: list[VectorRecord]) -> int:
        """Alias conforming to embeddings VectorStoreProtocol."""
        return await self.upsert_vectors(records)

    async def similarity_search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Perform cosine similarity search across in-memory vectors."""
        candidates: list[RetrievalResult] = []

        for record in self._records.values():
            if filter_metadata:
                skip = False
                for k, v in filter_metadata.items():
                    if k == "document_id":
                        if str(record.document_id) != str(v):
                            skip = True
                            break
                    else:
                        if record.metadata.get(k) != v:
                            skip = True
                            break
                if skip:
                    continue

            sim = _cosine_similarity(query_vector, record.vector)
            # Normalize [-1.0, 1.0] to [0.0, 1.0]
            score = max(0.0, min(1.0, (sim + 1.0) / 2.0))

            if score_threshold is not None and score < score_threshold:
                continue

            chunk_uuid = (
                record.id if isinstance(record.id, uuid.UUID) else uuid.UUID(str(record.id))
            )
            doc_uuid = (
                record.document_id
                if isinstance(record.document_id, uuid.UUID)
                else uuid.UUID(str(record.document_id))
                if record.document_id
                else uuid.uuid4()
            )

            candidates.append(
                RetrievalResult(
                    chunk_id=chunk_uuid,
                    document_id=doc_uuid,
                    text=record.metadata.get("content", ""),
                    score=score,
                    dense_score=score,
                    sparse_score=None,
                    metadata=dict(record.metadata),
                )
            )

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:top_k]

    async def delete_vectors(self, chunk_ids: list[uuid.UUID]) -> int:
        """Delete vectors by chunk IDs."""
        count = 0
        for cid in chunk_ids:
            key = str(cid)
            if key in self._records:
                del self._records[key]
                count += 1
        return count

    async def delete_by_document(self, document_id: uuid.UUID) -> int:
        """Delete all vectors belonging to a document."""
        doc_str = str(document_id)
        to_delete = [cid for cid, r in self._records.items() if str(r.document_id) == doc_str]
        for cid in to_delete:
            del self._records[cid]
        return len(to_delete)

    async def get_vector(self, chunk_id: uuid.UUID) -> VectorRecord | None:
        """Fetch a stored vector record by chunk ID."""
        return self._records.get(str(chunk_id))

    async def count(self) -> int:
        """Return total count of indexed vectors."""
        return len(self._records)
