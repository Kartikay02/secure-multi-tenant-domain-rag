"""Factory for instantiating LLM provider implementations."""

from app.core.config import LLMSettings, get_settings
from app.core.logging import get_logger
from app.rag.generation.interfaces import LLMProviderProtocol
from app.rag.generation.mock import MockLLMProvider
from app.rag.generation.openai import OpenAILLMProvider

logger = get_logger("app.rag.generation.factory")


class LLMFactory:
    """Factory for creating LLMProviderProtocol instances."""

    @staticmethod
    def create_provider(settings: LLMSettings | None = None) -> LLMProviderProtocol:
        """Create and return an LLM provider based on configuration."""
        llm_settings = settings or get_settings().llm
        provider = llm_settings.provider.lower()

        if provider == "mock":
            logger.info("Initializing MockLLMProvider", extra={"model": llm_settings.model})
            return MockLLMProvider(model_name=llm_settings.model)

        if provider == "openai":
            logger.info(
                "Initializing OpenAILLMProvider",
                extra={
                    "model": llm_settings.model,
                    "api_base": llm_settings.api_base,
                    "timeout_seconds": llm_settings.timeout_seconds,
                },
            )
            return OpenAILLMProvider(
                api_key=llm_settings.api_key,
                model=llm_settings.model,
                api_base=llm_settings.api_base,
                timeout_seconds=llm_settings.timeout_seconds,
                max_retries=llm_settings.max_retries,
                backoff_factor=llm_settings.backoff_factor,
                max_backoff_seconds=llm_settings.max_backoff_seconds,
            )

        raise ValueError(f"Unsupported LLM provider: '{provider}'")
