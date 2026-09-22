"""Unit tests for DeterministicOfflineTokenizer and offline encoding fallback (Issue 3)."""

from unittest.mock import patch

from app.core.tokenizer import (
    DeterministicOfflineTokenizer,
    get_cached_encoding,
)
from app.rag.chunking.recursive import RecursiveTokenChunker


def test_deterministic_offline_tokenizer_encode_decode() -> None:
    """Verify DeterministicOfflineTokenizer produces deterministic tokens and reconstructs text."""
    tok = DeterministicOfflineTokenizer(chars_per_token=4)
    text = "The quick brown fox jumps over the lazy dog."

    tokens = tok.encode(text)
    assert isinstance(tokens, list)
    assert len(tokens) > 0

    reconstructed = tok.decode(tokens)
    assert reconstructed == text


def test_deterministic_offline_tokenizer_subslice_decode() -> None:
    """Verify sub-slice of tokens can be decoded cleanly."""
    tok = DeterministicOfflineTokenizer(chars_per_token=4)
    text = "abcdefghijklmnop"
    tokens = tok.encode(text)

    # 16 chars / 4 = 4 tokens
    assert len(tokens) == 4

    slice_1 = tok.decode(tokens[0:2])
    assert slice_1 == "abcdefgh"

    slice_2 = tok.decode(tokens[2:4])
    assert slice_2 == "ijklmnop"


def test_deterministic_offline_tokenizer_empty_and_whitespace() -> None:
    """Verify empty text produces empty tokens and empty decode."""
    tok = DeterministicOfflineTokenizer(chars_per_token=4)
    assert tok.encode("") == []
    assert tok.decode([]) == ""
    assert tok.count_tokens("") == 0


def test_get_cached_encoding_force_offline() -> None:
    """Verify force_offline returns DeterministicOfflineTokenizer directly."""
    enc = get_cached_encoding(force_offline=True)
    assert isinstance(enc, DeterministicOfflineTokenizer)


def test_get_cached_encoding_offline_on_exception() -> None:
    """Verify exception in tiktoken triggers DeterministicOfflineTokenizer fallback without retry."""
    with patch("tiktoken.get_encoding", side_effect=Exception("Offline - No internet")):
        enc = get_cached_encoding("non_existent_or_offline_encoding_model")
        assert isinstance(enc, DeterministicOfflineTokenizer)
        tokens = enc.encode("Hello world")
        assert len(tokens) > 0


def test_recursive_chunker_with_offline_tokenizer() -> None:
    """Verify RecursiveTokenChunker operates completely offline without network."""
    with patch("tiktoken.get_encoding", side_effect=Exception("Offline - No network")):
        # Fresh encoding name to bypass cache
        chunker = RecursiveTokenChunker(
            chunk_size=20,
            chunk_overlap=5,
            encoding_name="offline_simulated_model_1",
        )
        assert isinstance(chunker.tokenizer, DeterministicOfflineTokenizer)

        sample = (
            "Section 1: Architecture Overview.\n\n"
            "This domain RAG system is engineered for resilient, multi-tenant, zero-internet runtime operations. "
            "Every component maintains deterministic fallbacks when external dependencies are unavailable."
        )
        chunks = chunker.split_text(sample)
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk.content) > 0
            assert chunk.token_count > 0


def test_tiktoken_package_unavailable_fallback() -> None:
    """Verify get_cached_encoding cleanly returns offline fallback when tiktoken is None."""
    with patch("app.core.tokenizer.tiktoken", None):
        enc = get_cached_encoding("cl100k_base_simulated_unavailable")
        assert isinstance(enc, DeterministicOfflineTokenizer)
        tokens = enc.encode("Sample test without tiktoken package installed")
        assert len(tokens) > 0
        assert enc.decode(tokens) == "Sample test without tiktoken package installed"


def test_context_builder_when_tiktoken_package_unavailable() -> None:
    """Verify ContextBuilder operates cleanly and budgets tokens when tiktoken is None."""
    import uuid

    from app.rag.context.builder import ContextBuilder
    from app.rag.vector.domain import RetrievalResult

    with patch("app.core.tokenizer.tiktoken", None):
        builder = ContextBuilder(max_tokens=100, max_chunks=3)
        assert isinstance(builder._tokenizer, DeterministicOfflineTokenizer)

        candidates = [
            RetrievalResult(
                chunk_id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                text="Kubernetes deployments require liveness and readiness probes for orchestrating zero-downtime rollouts.",
                score=0.95,
                metadata={"title": "K8s Guide", "page": 1},
            ),
            RetrievalResult(
                chunk_id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                text="Service mesh architecture provides mutual TLS encryption and traffic observability across pods.",
                score=0.90,
                metadata={"title": "Service Mesh", "page": 2},
            ),
        ]

        context = builder.build_context(
            candidates=candidates, query="What probes does Kubernetes need?"
        )
        assert len(context.documents) >= 1
        assert context.total_tokens <= 100
        assert len(context.formatted_context) > 0


def test_recursive_chunker_when_tiktoken_package_unavailable() -> None:
    """Verify RecursiveTokenChunker chunks text deterministically when tiktoken package is None."""
    with patch("app.core.tokenizer.tiktoken", None):
        chunker = RecursiveTokenChunker(chunk_size=30, chunk_overlap=5)
        assert isinstance(chunker.tokenizer, DeterministicOfflineTokenizer)
        text = "Deep technical audit of Domain RAG system ensures production zero-downtime reliability."
        chunks = chunker.split_text(text)
        assert len(chunks) >= 1
        assert all(c.token_count <= 30 for c in chunks)
