"""Base SQLAlchemy repository implementation with transaction and error handling."""

import uuid
from typing import TypeVar

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import DatabaseError, DuplicateEntityError
from app.core.logging import get_logger
from app.models.base import Base

logger = get_logger("app.repository")
T = TypeVar("T", bound=Base)


class BaseSQLAlchemyRepository[T]:
    """Reusable async CRUD operations for SQLAlchemy declarative models."""

    def __init__(self, session: AsyncSession, entity_class: type[T]) -> None:
        self.session = session
        self.entity_class = entity_class

    async def get_by_id(self, id: uuid.UUID) -> T | None:
        """Fetch entity by primary key UUID."""
        try:
            query = select(self.entity_class).where(self.entity_class.id == id)  # type: ignore[attr-defined]
            result = await self.session.execute(query)
            return result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            logger.error(f"Failed to fetch {self.entity_class.__name__} by ID {id}: {exc}")
            raise DatabaseError(f"Error fetching {self.entity_class.__name__}: {exc}") from exc

    async def list(self, offset: int = 0, limit: int = 50) -> list[T]:
        """Fetch a paginated list of entities."""
        try:
            query = (
                select(self.entity_class)
                .order_by(self.entity_class.created_at.desc())  # type: ignore[attr-defined]
                .offset(offset)
                .limit(limit)
            )
            result = await self.session.execute(query)
            return list(result.scalars().all())
        except SQLAlchemyError as exc:
            logger.error(f"Failed to list {self.entity_class.__name__} records: {exc}")
            raise DatabaseError(f"Error listing {self.entity_class.__name__}: {exc}") from exc

    async def count(self) -> int:
        """Count total records for this entity."""
        try:
            query = select(func.count(self.entity_class.id))  # type: ignore[attr-defined]
            result = await self.session.execute(query)
            return int(result.scalar_one() or 0)
        except SQLAlchemyError as exc:
            logger.error(f"Failed to count {self.entity_class.__name__} records: {exc}")
            raise DatabaseError(f"Error counting {self.entity_class.__name__}: {exc}") from exc

    async def create(self, entity: T) -> T:
        """Persist a single entity, catching duplicate constraint violations."""
        try:
            self.session.add(entity)
            await self.session.flush()
            return entity
        except IntegrityError as exc:
            await self.session.rollback()
            logger.warning(f"Integrity violation creating {self.entity_class.__name__}: {exc}")
            raise DuplicateEntityError(
                entity_name=self.entity_class.__name__,
                identifier=str(getattr(entity, "id", "unknown")),
            ) from exc
        except SQLAlchemyError as exc:
            await self.session.rollback()
            logger.error(f"Database error creating {self.entity_class.__name__}: {exc}")
            raise DatabaseError(f"Could not persist {self.entity_class.__name__}: {exc}") from exc

    async def delete(self, id: uuid.UUID) -> bool:
        """Delete an entity by its UUID."""
        entity = await self.get_by_id(id)
        if not entity:
            return False
        try:
            await self.session.delete(entity)
            await self.session.flush()
            return True
        except SQLAlchemyError as exc:
            await self.session.rollback()
            logger.error(f"Failed to delete {self.entity_class.__name__} ID {id}: {exc}")
            raise DatabaseError(f"Could not delete {self.entity_class.__name__}: {exc}") from exc
