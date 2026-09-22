"""Factory for instantiating reranker implementations based on configuration."""

from app.core.config import RetrievalSettings, get_settings
from app.core.logging import get_logger
from app.rag.reranking.cohere import CohereReranker
from app.rag.reranking.interfaces import RerankerProtocol
from app.rag.reranking.mock import MockReranker
from app.rag.reranking.noop import NoOpReranker

logger = get_logger("app.rag.reranking.factory")


class RerankerFactory:
    """Factory for creating cross-encoder reranker instances."""

    @staticmethod
    def create_reranker(settings: RetrievalSettings | None = None) -> RerankerProtocol:
        """Create a reranker instance conforming to RerankerProtocol.

        Args:
            settings: Optional retrieval settings. If None, loaded from global settings.

        Returns:
            RerankerProtocol implementation.
        """
        retrieval_settings = settings or get_settings().retrieval

        if (
            not retrieval_settings.reranker_enabled
            or retrieval_settings.reranker_provider == "none"
        ):
            logger.info("Reranker is disabled; initializing NoOpReranker pass-through.")
            return NoOpReranker()

        provider = retrieval_settings.reranker_provider.lower()

        if provider == "mock":
            logger.info(
                "Initializing MockReranker",
                extra={"model": retrieval_settings.reranker_model},
            )
            return MockReranker(model_name=retrieval_settings.reranker_model)

        if provider == "cohere":
            logger.info(
                "Initializing CohereReranker",
                extra={
                    "model": retrieval_settings.reranker_model,
                    "api_base": retrieval_settings.reranker_api_base,
                    "timeout_seconds": retrieval_settings.reranker_timeout_seconds,
                },
            )
            return CohereReranker(
                api_key=retrieval_settings.reranker_api_key,
                model=retrieval_settings.reranker_model,
                api_base=retrieval_settings.reranker_api_base,
                timeout_seconds=retrieval_settings.reranker_timeout_seconds,
            )

        if provider == "flashrank":
            raise NotImplementedError(
                "FlashRank local cross-encoder provider is planned for a future release. "
                "Please configure 'mock' or 'cohere'."
            )

        raise ValueError(f"Unsupported reranker provider: '{provider}'")
