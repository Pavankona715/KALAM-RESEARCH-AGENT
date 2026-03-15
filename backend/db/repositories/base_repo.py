"""
Base Repository
===============
Generic CRUD operations that all repositories inherit.
Repositories are the only place SQL queries live — routes never touch SQLAlchemy.

Pattern: Repository receives an AsyncSession from the route via dependency injection.
It never creates sessions itself — that's the caller's responsibility.
"""

import uuid
from typing import Any, Generic, TypeVar

from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.base import BaseModel

ModelType = TypeVar("ModelType", bound=BaseModel)


class BaseRepository(Generic[ModelType]):
    """
    Generic repository providing standard CRUD operations.

    Usage:
        class UserRepository(BaseRepository[User]):
            model = User

        user_repo = UserRepository()
        user = await user_repo.get_by_id(db, user_id)
    """

    model: type[ModelType]

    async def get_by_id(
        self, db: AsyncSession, id: uuid.UUID
    ) -> ModelType | None:
        """Fetch a single record by primary key."""
        result = await db.execute(
            select(self.model).where(self.model.id == id)
        )
        return result.scalar_one_or_none()

    async def get_all(
        self,
        db: AsyncSession,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> list[ModelType]:
        """Fetch all records with pagination."""
        result = await db.execute(
            select(self.model).offset(skip).limit(limit)
        )
        return list(result.scalars().all())

    async def create(self, db: AsyncSession, **kwargs: Any) -> ModelType:
        """Create and persist a new record."""
        instance = self.model(**kwargs)
        db.add(instance)
        await db.flush()   # Get the ID without committing
        await db.refresh(instance)
        return instance

    async def update(
        self,
        db: AsyncSession,
        id: uuid.UUID,
        **kwargs: Any,
    ) -> ModelType | None:
        """Update fields on an existing record."""
        await db.execute(
            update(self.model)
            .where(self.model.id == id)
            .values(**kwargs)
        )
        return await self.get_by_id(db, id)

    async def delete(self, db: AsyncSession, id: uuid.UUID) -> bool:
        """Delete a record by ID. Returns True if deleted, False if not found."""
        result = await db.execute(
            delete(self.model).where(self.model.id == id)
        )
        return result.rowcount > 0

    async def count(self, db: AsyncSession) -> int:
        """Count total records."""
        result = await db.execute(
            select(func.count()).select_from(self.model)
        )
        return result.scalar_one()

    async def exists(self, db: AsyncSession, id: uuid.UUID) -> bool:
        """Check if a record exists without loading it."""
        result = await db.execute(
            select(func.count())
            .select_from(self.model)
            .where(self.model.id == id)
        )
        return result.scalar_one() > 0