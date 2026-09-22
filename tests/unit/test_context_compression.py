"""Unit tests for context compression strategies."""

from app.rag.context.compression import (
    ExtractiveQueryCompression,
    NoOpCompression,
    WhitespaceNormalizerCompression,
)


def test_noop_compression() -> None:
    text = "  Exact text \n with formatting   "
    comp = NoOpCompression()
    assert comp.compress(text, "query") == text


def test_whitespace_normalizer_compression() -> None:
    text = "Line 1   has  extra spaces.\n\n\n\nLine 2 after big gap.\t\tTab here."
    comp = WhitespaceNormalizerCompression()
    result = comp.compress(text, "query")

    assert "\n\n\n" not in result
    assert "Line 1 has extra spaces." in result
    assert "Line 2 after big gap. Tab here." in result


def test_extractive_query_compression() -> None:
    text = (
        "The sky is blue and sunny. "
        "PostgreSQL WAL physical replication ensures standby nodes stay synchronized. "
        "Birds are flying in the park. "
        "Standby promotion allows zero data loss during unplanned database failover."
    )
    query = "PostgreSQL replication standby failover"

    comp = ExtractiveQueryCompression(min_sentences=1, max_sentences=2)
    compressed = comp.compress(text, query)

    # Sentences 2 and 4 should be retained because they match query keywords
    assert "PostgreSQL WAL physical replication" in compressed
    assert "Standby promotion allows zero data loss" in compressed
    # Irrelevant sentences 1 and 3 should be pruned
    assert "sky is blue" not in compressed
    assert "Birds are flying" not in compressed


def test_extractive_query_compression_empty_or_no_match() -> None:
    comp = ExtractiveQueryCompression(max_sentences=1)
    assert comp.compress("", "query") == ""

    text = "First sentence of general context. Second sentence with detail."
    # When query terms don't match, preserves initial sentence
    result = comp.compress(text, "xyz123")
    assert result == "First sentence of general context."
