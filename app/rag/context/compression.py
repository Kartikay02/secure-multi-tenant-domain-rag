"""Context compression and noise reduction strategies."""

import re

from app.rag.context.interfaces import CompressionStrategyProtocol


class NoOpCompression(CompressionStrategyProtocol):
    """Lossless pass-through compression preserving original chunk text exactly."""

    def compress(self, text: str, query: str) -> str:
        return text


class WhitespaceNormalizerCompression(CompressionStrategyProtocol):
    """Normalizes excessive whitespace, blank lines, and formatting noise."""

    def compress(self, text: str, query: str) -> str:
        if not text:
            return ""

        # Normalize line endings
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse 3+ newlines into 2
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        # Collapse multiple horizontal spaces and tabs into a single space
        normalized = re.sub(r"[ \t]+", " ", normalized)
        # Strip whitespace on individual lines
        lines = [line.strip() for line in normalized.split("\n")]
        return "\n".join(lines).strip()


class ExtractiveQueryCompression(CompressionStrategyProtocol):
    """Prunes query-irrelevant sentences from chunks while retaining core salient context.

    Splits text into sentences, scores each sentence by query term overlap,
    and retains sentences that match query terms or possess high salience,
    preserving context density.
    """

    def __init__(self, min_sentences: int = 1, max_sentences: int = 5) -> None:
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences

    def compress(self, text: str, query: str) -> str:
        if not text or not text.strip():
            return ""

        clean_text = text.strip()
        query_terms = set(re.findall(r"\w+", query.lower()))

        # If query has no alphanumeric terms, return clean text
        if not query_terms:
            return clean_text

        # Split into sentences
        raw_sentences = re.split(r"(?<=[.!?\n])\s+", clean_text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]

        if len(sentences) <= self.min_sentences:
            return clean_text

        scored_sentences: list[tuple[int, float, str]] = []
        for idx, sentence in enumerate(sentences):
            words = set(re.findall(r"\w+", sentence.lower()))
            overlap = len(query_terms.intersection(words))
            # Fraction of query terms covered by this sentence
            overlap_score = overlap / len(query_terms) if query_terms else 0.0
            scored_sentences.append((idx, overlap_score, sentence))

        # Filter sentences with positive overlap
        matching_sentences = [item for item in scored_sentences if item[1] > 0.0]

        # If no sentences explicitly match query terms, preserve initial sentences
        if not matching_sentences:
            return " ".join(sentences[: self.max_sentences])

        # Sort matching sentences by overlap score descending, take top max_sentences
        top_matches = sorted(matching_sentences, key=lambda x: x[1], reverse=True)[
            : self.max_sentences
        ]

        # Restore original reading order by sorting by idx
        top_matches.sort(key=lambda x: x[0])

        return " ".join(item[2] for item in top_matches)
