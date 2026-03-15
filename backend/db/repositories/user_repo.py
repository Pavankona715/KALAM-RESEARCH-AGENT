"""
User Repository
===============
Database operations for user management and authentication.
"""

import secrets
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models.user import User
from backend.db.repositories.base_repo import BaseRepository


class UserRepository(BaseRepository[User]):
    model = User

    async def get_by_email(
        self, db: AsyncSession, email: str
    ) -> User | None:
        result = await db.execute(
            select(User).where(User.email == email.lower().strip())
        )
        return result.scalar_one_or_none()

    async def get_by_api_key(
        self, db: AsyncSession, api_key: str
    ) -> User | None:
        result = await db.execute(
            select(User).where(User.api_key == api_key)
        )
        return result.scalar_one_or_none()

    async def create_user(
        self,
        db: AsyncSession,
        email: str,
        hashed_password: str | None = None,
    ) -> User:
        """Create a new user with a generated API key."""
        api_key = secrets.token_urlsafe(32)  # 256-bit random key
        return await self.create(
            db,
            email=email.lower().strip(),
            hashed_password=hashed_password,
            api_key=api_key,
        )

    async def rotate_api_key(
        self, db: AsyncSession, user_id: uuid.UUID
    ) -> str:
        """Generate and set a new API key for a user."""
        new_key = secrets.token_urlsafe(32)
        await self.update(db, user_id, api_key=new_key)
        return new_key


# Module-level singleton
user_repo = UserRepository()