"""Unit tests for TextNormalizer."""

from app.rag.ingestion.normalizer import TextNormalizer


def test_text_normalizer_whitespace_and_newlines() -> None:
    """Verify carriage returns are converted and excessive newlines collapsed."""
    raw = "Line 1   with   spaces\r\n\r\n\r\n\r\nLine 2\r\n\tLine 3"
    res = TextNormalizer.normalize(raw)

    assert "Line 1 with spaces\n\nLine 2\nLine 3" == res.text
    assert res.word_count == 8
    assert res.char_count > 0
    assert res.estimated_token_count > 0


def test_text_normalizer_unicode_nfkc() -> None:
    """Verify Unicode ligatures are decomposed into canonical characters."""
    raw = "The \ufb01rst \ufb02ight."  # 'fi' ligature and 'fl' ligature
    res = TextNormalizer.normalize(raw)
    assert res.text == "The first flight."


def test_text_normalizer_control_characters() -> None:
    """Verify ASCII control characters are stripped."""
    raw = "Clean\x07 text\x1b without\x00 controls."
    res = TextNormalizer.normalize(raw)
    assert res.text == "Clean text without controls."


def test_text_normalizer_empty() -> None:
    """Verify empty text produces 0 counts."""
    res = TextNormalizer.normalize("")
    assert res.text == ""
    assert res.word_count == 0
    assert res.char_count == 0
