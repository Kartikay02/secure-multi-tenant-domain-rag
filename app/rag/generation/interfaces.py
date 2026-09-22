"""Protocol contracts for LLM providers and prompt builders."""

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from app.rag.context.domain import AssembledContext
from app.rag.generation.domain import GeneratedResponse, GenerationParameters


@runtime_checkable
class PromptBuilderProtocol(Protocol):
    """Protocol for RAG prompt formatters."""

    def build_prompt(
        self,
        query: str,
        context: AssembledContext,
    ) -> tuple[str, str]:
        """Construct the system prompt and user prompt pair.

        Args:
            query: The user query string.
            context: AssembledContext containing formatted chunks and metadata.

        Returns:
            Tuple of (system_prompt, user_prompt).
        """
        ...


@runtime_checkable
class LLMProviderProtocol(Protocol):
    """Protocol for hosted or local LLM provider adapters."""

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        parameters: GenerationParameters | None = None,
        structured_schema: type[BaseModel] | None = None,
    ) -> GeneratedResponse:
        """Generate a complete text completion or structured response.

        Args:
            prompt: User prompt content.
            system_prompt: Optional grounding system prompt.
            parameters: Generation hyperparameters (temperature, max_tokens, etc.).
            structured_schema: Optional Pydantic model enforcing JSON structured output.

        Returns:
            GeneratedResponse containing answer, citations, confidence, and metadata.
        """
        ...

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        parameters: GenerationParameters | None = None,
    ) -> AsyncIterator[str]:
        """Yield text token deltas incrementally as they arrive from the provider.

        Args:
            prompt: User prompt content.
            system_prompt: Optional grounding system prompt.
            parameters: Generation hyperparameters.

        Yields:
            Token text deltas as strings.
        """
        ...
