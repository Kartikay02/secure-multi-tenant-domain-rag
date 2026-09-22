"""Domain models for vector retrieval."""

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievalResult:
    """Normalized domain representation of a retrieved document chunk."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    text: str
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
