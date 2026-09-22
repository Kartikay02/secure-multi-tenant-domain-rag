"""Protocol contracts for query preprocessing and normalization."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class QueryPreprocessorProtocol(Protocol):
    """Protocol for transforming and validating user search queries prior to retrieval."""

    def preprocess(self, query: str) -> str:
        """Sanitize, normalize, and validate a user query string.

        Args:
            query: Raw user query input.

        Returns:
            Cleaned and normalized query string.

        Raises:
            QueryValidationError: If query violates validation constraints (e.g. empty, length).
        """
        ...
