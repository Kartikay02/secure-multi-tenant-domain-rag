"""Configurable recursive token chunker for production RAG pipelines."""

import hashlib
import re
from typing import Any

from app.core.logging import get_logger
from app.core.tokenizer import get_cached_encoding
from app.rag.chunking.interfaces import ChunkerProtocol, ChunkPayload

logger = get_logger("app.rag.chunking")

DEFAULT_SEPARATORS: list[str] = [
    "\n\n",  # Paragraph boundaries
    "\n",  # Line breaks
    ". ",  # Sentence periods
    "? ",  # Question sentences
    "! ",  # Exclamation sentences
    "; ",  # Semicolons
    ", ",  # Commas
    " ",  # Words
    "",  # Character fallback
]


class RecursiveTokenChunker(ChunkerProtocol):
    """Semantic recursive character & token chunker.

    Splits text hierarchically, ensures deterministic bounds, enforces
    min/max chunk sizes, and preserves section/page metadata.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_length: int | None = None,
        max_chunk_length: int | None = None,
        separators: list[str] | None = None,
        encoding_name: str = "cl100k_base",
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})."
            )

        resolved_min = (
            min_chunk_length if min_chunk_length is not None else min(30, max(1, chunk_size // 5))
        )
        resolved_max = (
            max_chunk_length
            if max_chunk_length is not None
            else max(chunk_size + chunk_overlap, int(chunk_size * 1.2))
        )

        if resolved_min >= chunk_size:
            raise ValueError(
                f"min_chunk_length ({resolved_min}) must be less than chunk_size ({chunk_size})."
            )
        if resolved_max < chunk_size:
            raise ValueError(
                f"max_chunk_length ({resolved_max}) cannot be less than chunk_size ({chunk_size})."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = resolved_min
        self.max_chunk_length = resolved_max
        self.separators = separators or list(DEFAULT_SEPARATORS)

        self.tokenizer = get_cached_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the configured tiktoken model."""
        if not text:
            return 0
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def _split_text_recursively(self, text: str, separators: list[str]) -> list[str]:
        """Hierarchically divide text into pieces smaller than chunk_size."""
        if not text or not text.strip():
            return []

        if self.count_tokens(text) <= self.chunk_size:
            return [text.strip()]

        # Find first separator present in text
        chosen_sep = None
        new_separators: list[str] = []
        for i, sep in enumerate(separators):
            if sep == "":
                break
            if sep in text:
                chosen_sep = sep
                new_separators = separators[i + 1 :]
                break

        if chosen_sep is None:
            # Fallback for unbreakable text with no natural separators: token stride
            stride = max(1, self.chunk_size - self.chunk_overlap)
            tokens = self.tokenizer.encode(text, disallowed_special=())
            pieces: list[str] = []
            for start in range(0, len(tokens), stride):
                tok_slice = tokens[start : start + self.chunk_size]
                sub = self.tokenizer.decode(tok_slice).strip()
                if sub:
                    pieces.append(sub)
                if start + self.chunk_size >= len(tokens):
                    break
            return pieces

        # Split using selected separator
        splits = text.split(chosen_sep)
        pieces = []
        for s in splits:
            s_clean = s.strip()
            if not s_clean:
                continue
            if self.count_tokens(s_clean) <= self.chunk_size:
                pieces.append(s_clean)
            else:
                pieces.extend(self._split_text_recursively(s_clean, new_separators))

        return pieces

    def _merge_splits_with_overlap(self, pieces: list[str]) -> list[str]:
        """Combine split pieces into cohesive chunks respecting overlap and size ceilings."""
        if not pieces:
            return []

        chunks: list[str] = []
        current_pieces: list[str] = []
        current_tokens = 0

        for piece in pieces:
            p_tokens = self.count_tokens(piece)
            if p_tokens >= self.chunk_size:
                if current_pieces:
                    chunks.append(" ".join(current_pieces).strip())
                    current_pieces = []
                    current_tokens = 0
                chunks.append(piece)
                continue

            if current_tokens + p_tokens > self.chunk_size:
                if current_pieces:
                    chunk_str = " ".join(current_pieces).strip()
                    if chunk_str:
                        chunks.append(chunk_str)

                    if self.chunk_overlap > 0:
                        all_toks = self.tokenizer.encode(chunk_str, disallowed_special=())
                        if len(all_toks) > self.chunk_overlap:
                            overlap_toks = all_toks[-self.chunk_overlap :]
                            overlap_str = self.tokenizer.decode(overlap_toks).strip()
                            current_pieces = [overlap_str] if overlap_str else []
                            current_tokens = len(overlap_toks) if overlap_str else 0
                        else:
                            current_pieces = [chunk_str]
                            current_tokens = len(all_toks)
                    else:
                        current_pieces = []
                        current_tokens = 0

            current_pieces.append(piece)
            current_tokens += p_tokens

        if current_pieces:
            tail = " ".join(current_pieces).strip()
            if tail:
                chunks.append(tail)

        # Post-processing: prevent tiny meaningless trailing chunks
        if len(chunks) > 1:
            last_chunk = chunks[-1]
            last_tokens = self.count_tokens(last_chunk)
            if last_tokens < self.min_chunk_length:
                penultimate = chunks[-2]
                merged = f"{penultimate} {last_chunk}".strip()
                if self.count_tokens(merged) <= self.max_chunk_length:
                    chunks[-2] = merged
                    chunks.pop()
                elif not re.search(r"\w", last_chunk):
                    # Discard if it contains no alphanumeric content
                    chunks.pop()

        # Hard clamp: ensure no chunk exceeds max_chunk_length under any circumstance
        final_chunks: list[str] = []
        for c in chunks:
            c_str = c.strip()
            if not c_str:
                continue
            if self.count_tokens(c_str) > self.max_chunk_length:
                toks = self.tokenizer.encode(c_str, disallowed_special=())
                c_str = self.tokenizer.decode(toks[: self.max_chunk_length]).strip()
            final_chunks.append(c_str)

        return final_chunks

    def split_text(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        start_index: int = 0,
    ) -> list[ChunkPayload]:
        """Split a raw text string into a list of ChunkPayload objects."""
        if not text or not text.strip():
            return []

        base_meta = metadata.copy() if metadata else {}
        pieces = self._split_text_recursively(text, self.separators)
        chunk_texts = self._merge_splits_with_overlap(pieces)

        payloads: list[ChunkPayload] = []
        for i, chunk_text in enumerate(chunk_texts):
            idx = start_index + i
            tokens = self.count_tokens(chunk_text)
            chars = len(chunk_text)
            chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

            chunk_meta = {
                **base_meta,
                "chunk_index": idx,
                "chunk_hash": chunk_hash,
                "token_count": tokens,
                "char_count": chars,
            }

            payloads.append(
                ChunkPayload(
                    content=chunk_text,
                    chunk_index=idx,
                    token_count=tokens,
                    char_count=chars,
                    metadata=chunk_meta,
                )
            )

        return payloads

    def split_sections(
        self,
        sections: list[dict[str, Any]],
        base_metadata: dict[str, Any],
    ) -> list[ChunkPayload]:
        """Split structured document sections preserving section/page boundaries."""
        if not sections:
            return []

        all_chunks: list[ChunkPayload] = []
        current_index = 0

        for section_idx, section in enumerate(sections):
            sec_title = section.get("title", f"Section {section_idx + 1}")
            sec_content = section.get("content", "").strip()

            if not sec_content:
                continue

            # Detect page numbers if format is "Page X"
            page_number = None
            page_match = re.search(r"Page\s+(\d+)", sec_title, re.IGNORECASE)
            if page_match:
                page_number = int(page_match.group(1))

            section_meta = {
                **base_metadata,
                "section_title": sec_title,
                "section_index": section_idx,
                "page_number": page_number,
            }

            sec_chunks = self.split_text(
                text=sec_content,
                metadata=section_meta,
                start_index=current_index,
            )

            all_chunks.extend(sec_chunks)
            current_index += len(sec_chunks)

        return all_chunks
