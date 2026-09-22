"""Unit tests for query preprocessing and validation."""

import pytest

from app.core.exceptions import QueryValidationError
from app.rag.query.preprocessor import NoOpQueryPreprocessor, StandardQueryPreprocessor


class TestStandardQueryPreprocessor:
    """Test suite for StandardQueryPreprocessor normalization and boundary validation."""

    def test_basic_whitespace_stripping(self) -> None:
        preprocessor = StandardQueryPreprocessor()
        result = preprocessor.preprocess("   What is PostgreSQL?   ")
        assert result == "What is PostgreSQL?"

    def test_whitespace_collapsing(self) -> None:
        preprocessor = StandardQueryPreprocessor()
        result = preprocessor.preprocess("What   is    vector    search?\t\tAnd  lexical?")
        assert result == "What is vector search? And lexical?"

    def test_newlines_collapsing(self) -> None:
        preprocessor = StandardQueryPreprocessor()
        result = preprocessor.preprocess("Line 1\n\n\n\n\nLine 2")
        assert result == "Line 1\n\nLine 2"

    def test_unicode_normalization_nfkc(self) -> None:
        preprocessor = StandardQueryPreprocessor()
        # Full-width latin letters: Ｗｈａｔ -> What
        full_width = "Ｗｈａｔ ｉｓ ＲＡＧ？"
        result = preprocessor.preprocess(full_width)
        assert result == "What is RAG?"

    def test_control_character_stripping(self) -> None:
        preprocessor = StandardQueryPreprocessor()
        # Null byte \x00, bell \x07, escape \x1b
        query_with_control = "What is\x00 deep\x07 learning\x1b?"
        result = preprocessor.preprocess(query_with_control)
        assert result == "What is deep learning?"

    def test_preserves_valid_punctuation_and_symbols(self) -> None:
        preprocessor = StandardQueryPreprocessor()
        query = "How does HNSW (M=16, ef_construction=64) compare to IVF-Flat at $0.05/hr?"
        result = preprocessor.preprocess(query)
        assert result == query

    def test_rejects_empty_query(self) -> None:
        preprocessor = StandardQueryPreprocessor()
        with pytest.raises(QueryValidationError) as exc_info:
            preprocessor.preprocess("")
        assert exc_info.value.error_code == "QUERY_VALIDATION_ERROR"
        assert "empty or whitespace-only" in exc_info.value.message

    def test_rejects_whitespace_only_query(self) -> None:
        preprocessor = StandardQueryPreprocessor()
        with pytest.raises(QueryValidationError) as exc_info:
            preprocessor.preprocess("   \t  \n  ")
        assert exc_info.value.error_code == "QUERY_VALIDATION_ERROR"

    def test_rejects_query_shorter_than_min_length(self) -> None:
        preprocessor = StandardQueryPreprocessor(min_length=5)
        with pytest.raises(QueryValidationError) as exc_info:
            preprocessor.preprocess("RAG")
        assert "shorter than minimum required length" in exc_info.value.message
        assert exc_info.value.details["min_length"] == 5

    def test_rejects_query_longer_than_max_length(self) -> None:
        preprocessor = StandardQueryPreprocessor(max_length=20)
        long_query = "This query exceeds the maximum allowed length limit."
        with pytest.raises(QueryValidationError) as exc_info:
            preprocessor.preprocess(long_query)
        assert "exceeds maximum permitted length" in exc_info.value.message
        assert exc_info.value.details["max_length"] == 20

    def test_invalid_init_parameters(self) -> None:
        with pytest.raises(ValueError, match="min_length must be at least 1"):
            StandardQueryPreprocessor(min_length=0)

        with pytest.raises(ValueError, match="must be >= min_length"):
            StandardQueryPreprocessor(min_length=50, max_length=10)

    def test_rejects_non_string_type(self) -> None:
        preprocessor = StandardQueryPreprocessor()
        with pytest.raises(QueryValidationError) as exc_info:
            preprocessor.preprocess(12345)  # type: ignore[arg-type]
        assert "Query must be a string" in exc_info.value.message


class TestNoOpQueryPreprocessor:
    """Test suite for NoOpQueryPreprocessor."""

    def test_passes_query_through(self) -> None:
        preprocessor = NoOpQueryPreprocessor()
        assert preprocessor.preprocess("  Hello world  ") == "Hello world"

    def test_rejects_empty(self) -> None:
        preprocessor = NoOpQueryPreprocessor(min_length=1)
        with pytest.raises(QueryValidationError):
            preprocessor.preprocess("   ")
