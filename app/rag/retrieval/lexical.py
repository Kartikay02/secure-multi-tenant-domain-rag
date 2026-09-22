"""Lexical / keyword retriever using PostgreSQL full-text search with cross-dialect fallback."""

import hashlib
import re
import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import DatabaseError
from app.core.logging import get_logger
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.version import DocumentVersion
from app.rag.retrieval.interfaces import RetrieverProtocol
from app.rag.vector.domain import RetrievalResult

logger = get_logger("app.rag.retrieval.lexical")


_LEXICAL_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "what",
    "which",
    "where",
    "when",
    "who",
    "how",
    "why",
    "this",
    "about",
    "tell",
    "me",
    "give",
    "can",
    "you",
    "does",
    "do",
}


def _compute_fallback_lexical_score(query: str, text: str) -> float:
    """Compute normalized keyword matching score [0.0, 1.0] for non-PostgreSQL dialects."""
    raw_terms = [w.lower() for w in re.findall(r"\w+", query) if len(w) >= 2]
    content_terms = [w for w in raw_terms if w not in _LEXICAL_STOPWORDS] or raw_terms
    if not content_terms:
        return 0.0
    text_lower = text.lower()
    matched = 0.0
    for t in content_terms:
        # Word boundary match
        matches = len(re.findall(r"\b" + re.escape(t) + r"\b", text_lower))
        if matches > 0:
            matched += 1.0 + min(0.5, (matches - 1) * 0.1)
    return min(1.0, round(matched / len(content_terms), 4))


class PGLexicalRetriever(RetrieverProtocol):
    """Retrieves document chunks using exact keyword matching and full-text search."""

    def __init__(self, session: AsyncSession | None = None) -> None:
        self._explicit_session = session

    @property
    def session(self) -> AsyncSession:
        if self._explicit_session is not None:
            return self._explicit_session
        from app.core.database import get_current_session

        current = get_current_session()
        if current is not None:
            return current
        raise DatabaseError("No active database session available for PGLexicalRetriever.")

    @session.setter
    def session(self, value: AsyncSession | None) -> None:
        self._explicit_session = value

    def _is_postgresql(self) -> bool:
        bind = self.session.bind
        if bind and hasattr(bind, "dialect"):
            return str(bind.dialect.name).lower() == "postgresql"
        return False

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Execute sparse lexical keyword search."""
        if not query or not query.strip():
            return []

        clean_query = query.strip()
        query_fp = hashlib.sha256(clean_query.encode("utf-8")).hexdigest()[:8]
        logger.debug(
            f"PGLexicalRetriever executing search for query_fp={query_fp} (len={len(clean_query)})",
            extra={"query_hash": query_fp, "query_length": len(clean_query)},
        )

        try:
            is_pg = self._is_postgresql()

            if is_pg:
                # 1. Native PostgreSQL Full-Text Search
                ts_query = func.plainto_tsquery("english", clean_query)
                ts_vector = func.to_tsvector("english", DocumentChunk.content)
                rank_expr = func.ts_rank_cd(ts_vector, ts_query)

                stmt = (
                    select(
                        DocumentChunk,
                        Document,
                        rank_expr.label("rank"),
                    )
                    .join(DocumentVersion, DocumentChunk.document_version_id == DocumentVersion.id)
                    .join(Document, DocumentVersion.document_id == Document.id)
                    .where(ts_vector.op("@@")(ts_query))
                )

                if filter_metadata:
                    if "document_id" in filter_metadata:
                        doc_uuid = uuid.UUID(str(filter_metadata["document_id"]))
                        stmt = stmt.where(Document.id == doc_uuid)
                    for k, v in filter_metadata.items():
                        if k != "document_id":
                            stmt = stmt.where(DocumentChunk.metadata_json.contains({k: v}))

                stmt = stmt.order_by(rank_expr.desc()).limit(top_k * 2)
                result = await self.session.execute(stmt)
                rows = result.all()

                results: list[RetrievalResult] = []
                for chunk, doc, raw_rank in rows:
                    rank_val = float(raw_rank) if raw_rank is not None else 0.0
                    # Normalize ts_rank_cd into [0.0, 1.0]
                    norm_score = min(1.0, round(rank_val / (rank_val + 0.1), 4))

                    if score_threshold is not None and norm_score < score_threshold:
                        continue

                    meta = dict(chunk.metadata_json or {})
                    meta["document_name"] = doc.name
                    meta["document_source"] = doc.source
                    meta["chunk_index"] = chunk.chunk_index

                    results.append(
                        RetrievalResult(
                            chunk_id=chunk.id,
                            document_id=doc.id,
                            text=chunk.content,
                            score=norm_score,
                            dense_score=None,
                            sparse_score=norm_score,
                            metadata=meta,
                        )
                    )

                results.sort(key=lambda r: r.score, reverse=True)
                return results[:top_k]

            else:
                # 2. SQLite / in-memory test fallback
                stmt = (
                    select(DocumentChunk, Document)
                    .join(DocumentVersion, DocumentChunk.document_version_id == DocumentVersion.id)
                    .join(Document, DocumentVersion.document_id == Document.id)
                )

                if filter_metadata and "document_id" in filter_metadata:
                    doc_uuid = uuid.UUID(str(filter_metadata["document_id"]))
                    stmt = stmt.where(Document.id == doc_uuid)

                result = await self.session.execute(stmt)
                rows = result.all()

                candidates: list[RetrievalResult] = []
                for chunk, doc in rows:
                    if filter_metadata:
                        skip = False
                        for k, v in filter_metadata.items():
                            if k == "document_id":
                                continue
                            if chunk.metadata_json.get(k) != v:
                                skip = True
                                break
                        if skip:
                            continue

                    score = _compute_fallback_lexical_score(clean_query, chunk.content)
                    is_summary_q = any(
                        w in clean_query.lower()
                        for w in [
                            "summarize",
                            "summary",
                            "overview",
                            "about this",
                            "about the",
                            "raw openapi",
                            "raw json",
                            "raw definition",
                            "raw schema",
                            "raw content",
                            "show me the raw",
                            "show raw",
                        ]
                    )
                    if is_summary_q:
                        if chunk.chunk_index == 0:
                            score = max(score, 0.85)
                        elif any(
                            h in chunk.content.lower()
                            for h in [
                                "overview",
                                "architecture",
                                "consolidated",
                                "introduction",
                                "readme",
                            ]
                        ):
                            score = max(score, 0.70)

                    if score <= 0.0:
                        continue

                    if score_threshold is not None and score < score_threshold:
                        continue

                    meta = dict(chunk.metadata_json or {})
                    meta["document_name"] = doc.name
                    meta["document_source"] = doc.source
                    meta["chunk_index"] = chunk.chunk_index

                    candidates.append(
                        RetrievalResult(
                            chunk_id=chunk.id,
                            document_id=doc.id,
                            text=chunk.content,
                            score=score,
                            dense_score=None,
                            sparse_score=score,
                            metadata=meta,
                        )
                    )

                candidates.sort(key=lambda r: r.score, reverse=True)
                return candidates[:top_k]

        except SQLAlchemyError as exc:
            logger.error(f"Error during lexical retrieval: {exc}")
            raise DatabaseError(f"Database error during lexical retrieval: {exc}") from exc
