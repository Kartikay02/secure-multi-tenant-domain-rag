"""Factory for instantiating configured embedding providers."""

import httpx

from app.core.config import EmbeddingSettings
from app.core.exceptions import ConfigurationError
from app.core.logging import get_logger
from app.rag.embeddings.interfaces import EmbeddingProtocol
from app.rag.embeddings.mock import MockEmbeddingProvider
from app.rag.embeddings.openai import OpenAIEmbeddingProvider

logger = get_logger("app.rag.embeddings.factory")


class EmbeddingProviderFactory:
    """Factory resolving embedding provider implementations based on settings."""

    @staticmethod
    def create(
        settings: EmbeddingSettings,
        client: httpx.AsyncClient | None = None,
    ) -> EmbeddingProtocol:
        """Create and configure an embedding provider instance.

        Args:
            settings: EmbeddingSettings containing provider type, model, dimension, credentials.
            client: Optional pre-configured httpx.AsyncClient.

        Returns:
            An instance conforming to EmbeddingProtocol.

        Raises:
            ConfigurationError: If the provider is unrecognized.
        """
        provider = settings.provider.lower()
        logger.info(f"Creating embedding provider: '{provider}' for model '{settings.model}'")

        if provider == "mock":
            return MockEmbeddingProvider(
                dimension=settings.dimension,
                model_name=settings.model,
                batch_size=settings.batch_size,
            )

        if provider == "openai":
            return OpenAIEmbeddingProvider(
                api_key=settings.api_key,
                api_base=settings.api_base,
                model=settings.model,
                dimension=settings.dimension,
                batch_size=settings.batch_size,
                timeout_seconds=settings.timeout_seconds,
                max_retries=settings.max_retries,
                backoff_factor=settings.backoff_factor,
                max_backoff_seconds=settings.max_backoff_seconds,
                client=client,
            )

        raise ConfigurationError(
            f"Unsupported embedding provider: '{settings.provider}'. Supported: 'mock', 'openai'."
        )
