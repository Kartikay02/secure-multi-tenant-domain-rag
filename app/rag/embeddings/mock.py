"""Deterministic mock embedding provider for testing and offline development."""

import asyncio
import hashlib
import math
import random

from app.core.exceptions import EmbeddingDimensionMismatchError, EmbeddingError
from app.core.logging import get_logger
from app.rag.embeddings.interfaces import EmbeddingProtocol

logger = get_logger("app.rag.embeddings.mock")


_MOCK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "what",
    "which",
    "where",
    "when",
    "who",
    "how",
    "why",
    "this",
    "about",
}


class MockEmbeddingProvider(EmbeddingProtocol):
    """Deterministic, L2-normalized pseudo-embedding provider.

    Generates reproducible unit vectors seeded from terms in input texts.
    Supports simulated transient failures, latency, and dimension validation for tests.
    """

    def __init__(
        self,
        dimension: int = 384,
        model_name: str = "mock-embedding-v1",
        batch_size: int = 32,
        simulate_failure: bool = False,
        failure_count: int = 0,
        simulate_latency: float = 0.0,
    ) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self._dimension = dimension
        self._model_name = model_name
        self.batch_size = batch_size
        self.simulate_failure = simulate_failure
        self.failure_count = failure_count
        self.simulate_latency = simulate_latency
        self.total_calls = 0
        self.embedded_text_count = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def _generate_vector(self, text: str) -> list[float]:
        """Generate a deterministic L2-normalized vector for a given text string.

        Uses term-hash composition with downweighted stopwords so that queries and
        semantically relevant chunks share directional components.
        """
        import re

        words = [w.lower() for w in re.findall(r"\w+", text)]
        if not words:
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
            rng = random.Random(seed)
            raw = [rng.gauss(0.0, 1.0) for _ in range(self._dimension)]
            norm = math.sqrt(sum(x * x for x in raw)) or 1.0
            return [round(x / norm, 7) for x in raw]

        acc = [0.0] * self._dimension
        for w in words:
            weight = 0.05 if w in _MOCK_STOPWORDS else 1.0
            seed = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16) % (2**32)
            rng = random.Random(seed)
            for i in range(self._dimension):
                acc[i] += rng.gauss(0.0, 1.0) * weight

        norm = math.sqrt(sum(x * x for x in acc)) or 1.0
        vec = [round(x / norm, 7) for x in acc]

        # Verify dimension
        if len(vec) != self._dimension:
            raise EmbeddingDimensionMismatchError(
                expected=self._dimension,
                received=len(vec),
                model=self._model_name,
            )
        return vec

    async def _handle_simulation(self) -> None:
        """Handle simulated latency or failure conditions."""
        self.total_calls += 1

        if self.simulate_latency > 0:
            await asyncio.sleep(self.simulate_latency)

        if self.simulate_failure:
            raise EmbeddingError("Simulated persistent embedding provider failure.")

        if self.failure_count > 0:
            self.failure_count -= 1
            logger.warning(
                f"Simulating transient embedding failure. Remaining failures: {self.failure_count}"
            )
            raise EmbeddingError("Simulated transient embedding failure.")

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding vector for a single query or text."""
        await self._handle_simulation()
        self.embedded_text_count += 1
        return self._generate_vector(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a list of documents in batches."""
        if not texts:
            return []

        results: list[list[float]] = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            await self._handle_simulation()
            for t in batch:
                results.append(self._generate_vector(t))
            self.embedded_text_count += len(batch)

        return results
