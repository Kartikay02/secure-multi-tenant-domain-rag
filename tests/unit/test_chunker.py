"""Unit tests for RecursiveTokenChunker covering all required edge cases."""

from app.rag.chunking.recursive import RecursiveTokenChunker


def test_short_document_single_chunk() -> None:
    """Verify short documents smaller than chunk_size produce exactly one chunk."""
    chunker = RecursiveTokenChunker(chunk_size=100, chunk_overlap=20, min_chunk_length=10)
    text = (
        "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python."
    )
    chunks = chunker.split_text(text, metadata={"doc_id": "123"})

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].content == text
    assert chunks[0].token_count < 100
    assert chunks[0].metadata["doc_id"] == "123"
    assert "chunk_hash" in chunks[0].metadata


def test_large_document_multiple_chunks() -> None:
    """Verify large documents are split into multiple chunks respecting max_chunk_length."""
    chunker = RecursiveTokenChunker(chunk_size=50, chunk_overlap=10, max_chunk_length=70)
    paragraphs = [
        f"Paragraph {i}: Production RAG architectures require robust chunking to preserve "
        f"semantic context while staying within embedding model token windows."
        for i in range(15)
    ]
    large_text = "\n\n".join(paragraphs)

    chunks = chunker.split_text(large_text)
    assert len(chunks) > 3

    for idx, c in enumerate(chunks):
        assert c.chunk_index == idx
        assert c.token_count <= 70  # strictly bounded by max_chunk_length
        assert len(c.content) > 0


def test_empty_and_whitespace_documents() -> None:
    """Verify empty or whitespace-only documents yield zero chunks."""
    chunker = RecursiveTokenChunker()
    assert chunker.split_text("") == []
    assert chunker.split_text("   \n\t  \n  ") == []
    assert chunker.split_sections([], base_metadata={}) == []


def test_repeated_text_deterministic() -> None:
    """Verify repeated sentences chunk deterministically without infinite loops."""
    chunker = RecursiveTokenChunker(chunk_size=60, chunk_overlap=10)
    repeated = "The quick brown fox jumps over the lazy dog. " * 30

    run_1 = chunker.split_text(repeated)
    run_2 = chunker.split_text(repeated)

    assert len(run_1) > 1
    assert len(run_1) == len(run_2)
    for c1, c2 in zip(run_1, run_2, strict=True):
        assert c1.content == c2.content
        assert c1.token_count == c2.token_count
        assert c1.metadata["chunk_hash"] == c2.metadata["chunk_hash"]


def test_overlapping_chunks() -> None:
    """Verify consecutive chunks preserve target overlap between chunk boundaries."""
    chunker = RecursiveTokenChunker(chunk_size=20, chunk_overlap=8, separators=["\n\n", "\n", " "])
    text = (
        "Alpha beta gamma delta epsilon zeta eta theta. "
        "Iota kappa lambda mu nu xi omicron pi rho sigma. "
        "Tau upsilon phi chi psi omega end of text."
    )
    chunks = chunker.split_text(text)
    assert len(chunks) >= 2

    # Verify that words at the end of chunk 0 appear at the beginning of chunk 1
    words_chunk_0 = set(chunks[0].content.split()[-4:])
    words_chunk_1 = set(chunks[1].content.split()[:8])
    overlap_intersection = words_chunk_0.intersection(words_chunk_1)
    assert len(overlap_intersection) > 0


def test_page_boundaries_metadata_preservation() -> None:
    """Verify section titles and page numbers are preserved across chunks."""
    chunker = RecursiveTokenChunker(chunk_size=50, chunk_overlap=10)
    sections = [
        {"title": "Page 1", "content": "Welcome to page one of the employee policy."},
        {
            "title": "Page 2",
            "content": "This is page two detailing remote work rules and guidelines.",
        },
    ]
    base_meta = {"source": "handbook.pdf", "version_id": "v1"}

    chunks = chunker.split_sections(sections, base_metadata=base_meta)
    assert len(chunks) == 2

    assert chunks[0].metadata["page_number"] == 1
    assert chunks[0].metadata["section_title"] == "Page 1"
    assert chunks[0].metadata["source"] == "handbook.pdf"

    assert chunks[1].metadata["page_number"] == 2
    assert chunks[1].metadata["section_title"] == "Page 2"
    assert chunks[1].chunk_index == 1


def test_malformed_unbreakable_text() -> None:
    """Verify unbreakable text (e.g. 2000 continuous characters without spaces) is safely hard-sliced."""
    chunker = RecursiveTokenChunker(chunk_size=50, chunk_overlap=10, max_chunk_length=60)
    unbreakable = "A" * 2000

    chunks = chunker.split_text(unbreakable)
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 60


def test_prevent_tiny_meaningless_chunks() -> None:
    """Verify tiny trailing sentences are merged with previous chunks rather than left isolated."""
    # min_chunk_length = 20 tokens
    chunker = RecursiveTokenChunker(
        chunk_size=50, chunk_overlap=10, min_chunk_length=20, max_chunk_length=80
    )
    text = (
        "This is a comprehensive paragraph discussing database design and transaction boundaries in PostgreSQL. "
        "It contains enough words to fill the main chunk target size. Short end."
    )
    chunks = chunker.split_text(text)

    # "Short end." should be merged into the previous chunk instead of standing alone
    for c in chunks:
        assert c.token_count >= 20 or len(chunks) == 1
