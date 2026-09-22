"""Base repository protocol interface."""

import uuid
from typing import Protocol, TypeVar

T = TypeVar("T")


class BaseRepositoryProtocol[T](Protocol):
    """Generic data access protocol for entity CRUD operations."""

    async def get_by_id(self, id: uuid.UUID) -> T | None:
        """Fetch a single entity by its primary key UUID."""
        ...

    async def list(self, offset: int = 0, limit: int = 50) -> list[T]:
        """Fetch a paginated list of entities."""
        ...

    async def count(self) -> int:
        """Return the total number of records."""
        ...

    async def create(self, entity: T) -> T:
        """Persist a new entity into the database."""
        ...

    async def delete(self, id: uuid.UUID) -> bool:
        """Delete an entity by its UUID."""
        ...
