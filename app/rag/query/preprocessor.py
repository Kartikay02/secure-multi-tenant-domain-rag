import hashlib
import re
import unicodedata

from app.core.exceptions import QueryValidationError
from app.core.logging import get_logger
from app.rag.query.interfaces import QueryPreprocessorProtocol

logger = get_logger("app.rag.query.preprocessor")


class StandardQueryPreprocessor(QueryPreprocessorProtocol):
    """Production query preprocessor applying Unicode normalization, control character removal,

    whitespace collapsing, and bounds validation.
    """

    def __init__(
        self,
        min_length: int = 2,
        max_length: int = 2000,
        normalize_unicode: bool = True,
        strip_control_chars: bool = True,
        collapse_whitespace: bool = True,
    ) -> None:
        if min_length < 1:
            raise ValueError(f"min_length must be at least 1, got {min_length}")
        if max_length < min_length:
            raise ValueError(f"max_length ({max_length}) must be >= min_length ({min_length})")

        self.min_length = min_length
        self.max_length = max_length
        self.normalize_unicode = normalize_unicode
        self.strip_control_chars = strip_control_chars
        self.collapse_whitespace = collapse_whitespace

    def preprocess(self, query: str) -> str:
        """Sanitize, normalize, and validate the input query string."""
        if not isinstance(query, str):
            raise QueryValidationError(
                "Query must be a string.",
                details={"received_type": type(query).__name__},
            )

        raw = query.strip()
        if not raw:
            raise QueryValidationError(
                "Query must not be empty or whitespace-only.",
                details={"raw_query": query},
            )

        cleaned = raw

        # 1. Unicode NFKC Normalization (e.g. compatibility characters, full-width)
        if self.normalize_unicode:
            cleaned = unicodedata.normalize("NFKC", cleaned)

        # 2. Strip Non-Printable and Control Characters (preserving standard whitespace)
        if self.strip_control_chars:
            cleaned = "".join(
                ch
                for ch in cleaned
                if ch in "\n\r\t" or not unicodedata.category(ch).startswith("C")
            )

        # 3. Collapse Redundant Whitespace
        if self.collapse_whitespace:
            # Collapse horizontal tabs and multiple spaces into a single space
            cleaned = re.sub(r"[ \t]+", " ", cleaned)
            # Collapse more than two consecutive newlines into double newline
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
            cleaned = cleaned.strip()

        # 4. Validate Boundaries
        if len(cleaned) < self.min_length:
            raise QueryValidationError(
                f"Query length ({len(cleaned)}) is shorter than minimum required length ({self.min_length}).",
                details={"query_length": len(cleaned), "min_length": self.min_length},
            )

        if len(cleaned) > self.max_length:
            raise QueryValidationError(
                f"Query length ({len(cleaned)}) exceeds maximum permitted length ({self.max_length}).",
                details={"query_length": len(cleaned), "max_length": self.max_length},
            )

        query_fp = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:8]
        logger.debug(
            f"Preprocessed query_fp={query_fp} (clean_length={len(cleaned)})",
            extra={"raw_length": len(query), "clean_length": len(cleaned), "query_hash": query_fp},
        )
        return cleaned


class NoOpQueryPreprocessor(QueryPreprocessorProtocol):
    """Pass-through query preprocessor performing minimal strip without structural transformations."""

    def __init__(self, min_length: int = 1) -> None:
        self.min_length = min_length

    def preprocess(self, query: str) -> str:
        """Strip leading/trailing whitespace and ensure non-empty."""
        if not isinstance(query, str):
            raise QueryValidationError(
                "Query must be a string.",
                details={"received_type": type(query).__name__},
            )

        cleaned = query.strip()
        if len(cleaned) < self.min_length:
            raise QueryValidationError(
                f"Query must not be empty (minimum length {self.min_length}).",
                details={"query_length": len(cleaned), "min_length": self.min_length},
            )
        return cleaned
