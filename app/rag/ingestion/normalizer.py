"""Text normalization and cleanup for parsed document content."""

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedText:
    """Represents cleaned and normalized text alongside statistical metrics."""

    text: str
    char_count: int
    word_count: int
    estimated_token_count: int


class TextNormalizer:
    """Applies canonical Unicode normalization, whitespace hygiene, and control char stripping."""

    @staticmethod
    def normalize(text: str) -> NormalizedText:
        """Normalize raw extracted text into canonical RAG-ready format."""
        if not text:
            return NormalizedText(text="", char_count=0, word_count=0, estimated_token_count=0)

        # 1. Unicode canonical decomposition & recomposition (NFKC)
        # Normalizes ligatures (e.g. 'fi' -> 'f' + 'i') and compatibility characters
        normalized = unicodedata.normalize("NFKC", text)

        # 2. Normalize carriage returns to standard linefeeds
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

        # 3. Strip non-printable ASCII control characters except \t and \n
        normalized = "".join(
            ch for ch in normalized if ch in ("\n", "\t") or unicodedata.category(ch)[0] != "C"
        )

        # 4. Strip trailing spaces from each line and collapse redundant spaces
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.split("\n")]

        # 5. Collapse excessive blank lines (>2) to standard paragraph breaks (2 newlines)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

        # 6. Calculate statistics
        words = cleaned_text.split()
        word_count = len(words)
        char_count = len(cleaned_text)
        # Heuristic: 1 word ~ 1.33 tokens in typical English technical corpora
        estimated_tokens = int(round(word_count * 1.33))

        return NormalizedText(
            text=cleaned_text,
            char_count=char_count,
            word_count=word_count,
            estimated_token_count=estimated_tokens,
        )
