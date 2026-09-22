"""Deterministic offline tokenizer and centralized encoding cache (Issue 3 / SEC-20 / SEC-45).

Guarantees zero runtime internet requirement for token counting, chunking, and context budgeting.
"""

from typing import Any

try:
    import tiktoken
except ImportError:
    tiktoken = None  # type: ignore[assignment]

from app.core.logging import get_logger

logger = get_logger("app.core.tokenizer")

_ENCODING_CACHE: dict[str, Any] = {}


def is_tiktoken_available() -> bool:
    """Return True if tiktoken package is installed and importable."""
    return tiktoken is not None


class DeterministicOfflineTokenizer:
    """Offline heuristic tokenizer estimating tokens when tiktoken encoding is unavailable.

    Splits text into deterministic ~4-character units, maintaining a bi-directional
    vocabulary so that encode() and decode() are fully reversible and consistent across slices.
    Guarantees zero external network dependencies.
    """

    def __init__(self, chars_per_token: int = 4) -> None:
        self.chars_per_token = max(1, chars_per_token)
        self._vocab: dict[str, int] = {}
        self._inv_vocab: dict[int, str] = {}
        self._next_id: int = 1

    def encode(self, text: str, *args: Any, **kwargs: Any) -> list[int]:
        """Convert string into deterministic sequence of integer token IDs."""
        if not text:
            return []
        token_ids: list[int] = []
        cpt = self.chars_per_token
        for i in range(0, len(text), cpt):
            segment = text[i : i + cpt]
            tok_id = self._vocab.get(segment)
            if tok_id is None:
                tok_id = self._next_id
                self._next_id += 1
                self._vocab[segment] = tok_id
                self._inv_vocab[tok_id] = segment
            token_ids.append(tok_id)
        return token_ids

    def decode(self, tokens: list[int], *args: Any, **kwargs: Any) -> str:
        """Reconstruct string from sequence of token IDs."""
        if not tokens:
            return ""
        return "".join(self._inv_vocab.get(t, "") for t in tokens)

    def count_tokens(self, text: str) -> int:
        """Return token count approximation for given string."""
        if not text:
            return 0
        return max(1, (len(text) + self.chars_per_token - 1) // self.chars_per_token)


_GLOBAL_OFFLINE_TOKENIZER = DeterministicOfflineTokenizer()


def get_cached_encoding(encoding_name: str = "cl100k_base", force_offline: bool = False) -> Any:
    """Retrieve cached tiktoken encoding with deterministic offline fallback.

    Never attempts internet-bound fallback retries when tiktoken fails.
    If tiktoken is unavailable, raises an exception, or force_offline is set,
    returns DeterministicOfflineTokenizer.
    """
    if force_offline or tiktoken is None:
        return _GLOBAL_OFFLINE_TOKENIZER

    if encoding_name in _ENCODING_CACHE:
        return _ENCODING_CACHE[encoding_name]

    try:
        enc = tiktoken.get_encoding(encoding_name)
        _ENCODING_CACHE[encoding_name] = enc
        return enc
    except Exception as exc:
        logger.warning(
            f"Unable to load tiktoken encoding '{encoding_name}' ({exc}). "
            f"Using deterministic offline fallback tokenizer."
        )
        fallback = DeterministicOfflineTokenizer()
        _ENCODING_CACHE[encoding_name] = fallback
        return fallback
