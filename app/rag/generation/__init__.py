"""LLM generation layer for grounded, citation-backed answer synthesis."""

from app.rag.generation.domain import (
    GeneratedResponse,
    GenerationParameters,
    StreamChunk,
    StructuredGenerationPayload,
)
from app.rag.generation.factory import LLMFactory
from app.rag.generation.interfaces import LLMProviderProtocol, PromptBuilderProtocol
from app.rag.generation.mock import MockLLMProvider
from app.rag.generation.openai import OpenAILLMProvider
from app.rag.generation.prompt import DEFAULT_RAG_SYSTEM_PROMPT, PromptBuilder
from app.rag.generation.service import GenerationService

__all__ = [
    "DEFAULT_RAG_SYSTEM_PROMPT",
    "GeneratedResponse",
    "GenerationParameters",
    "GenerationService",
    "LLMFactory",
    "LLMProviderProtocol",
    "MockLLMProvider",
    "OpenAILLMProvider",
    "PromptBuilder",
    "PromptBuilderProtocol",
    "StreamChunk",
    "StructuredGenerationPayload",
]
