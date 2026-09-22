"""Context assembly subsystem for formatting, deduplicating, and bounding prompt context."""

from app.rag.context.builder import ContextBuilder
from app.rag.context.compression import (
    ExtractiveQueryCompression,
    NoOpCompression,
    WhitespaceNormalizerCompression,
)
from app.rag.context.deduplication import ContentDeduplicator
from app.rag.context.domain import AssembledContext, ContextDocument
from app.rag.context.formatting import ContextFormatter
from app.rag.context.interfaces import (
    CompressionStrategyProtocol,
    ContextBuilderProtocol,
    DeduplicatorProtocol,
    OrderingStrategyProtocol,
)
from app.rag.context.ordering import (
    DocumentOrderOrdering,
    LostInTheMiddleOrdering,
    RelevanceOrdering,
)

__all__ = [
    "AssembledContext",
    "CompressionStrategyProtocol",
    "ContentDeduplicator",
    "ContextBuilder",
    "ContextBuilderProtocol",
    "ContextDocument",
    "ContextFormatter",
    "DeduplicatorProtocol",
    "DocumentOrderOrdering",
    "ExtractiveQueryCompression",
    "LostInTheMiddleOrdering",
    "NoOpCompression",
    "OrderingStrategyProtocol",
    "RelevanceOrdering",
    "WhitespaceNormalizerCompression",
]
