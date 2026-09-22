"""Abstract embedding protocol and DTO definitions."""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class EmbeddingResult:
    """Encapsulates generated embedding vector and diagnostic metadata."""

    vector: list[float]
    dimension: int
    token_count: int = 0
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class EmbeddingProtocol(Protocol):
    """Abstract interface for dense text embedding models and services.

    Any provider (local ONNX/FastEmbed, hosted OpenAI/vLLM/TEI, or mock)
    must implement this protocol.
    """

    @property
    def dimension(self) -> int:
        """Dimensionality of output vectors (e.g. 384, 768, 1536)."""
        ...

    @property
    def model_name(self) -> str:
        """Name or identifier of the underlying model."""
        ...

    async def embed_text(self, text: str) -> list[float]:
        """Generate a dense vector embedding for a single query or text snippet.

        Args:
            text: Input string to embed.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            EmbeddingError: If generation fails or returns invalid dimensions.
        """
        ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate dense vector embeddings for a list of document chunks.

        Handles batching transparently and preserves original input ordering.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of vector embeddings corresponding 1-to-1 with input texts.

        Raises:
            EmbeddingError: If generation fails or returns invalid dimensions.
        """
        ...
