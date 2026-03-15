"""
Seed Data Script
================
Creates a default admin user with an API key for development.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.db.session import init_db, get_db_context
from backend.db.repositories.user_repo import user_repo


async def seed():
    # Must initialize the DB engine first (mirrors FastAPI lifespan startup)
    await init_db()

    async with get_db_context() as db:
        existing = await user_repo.get_by_email(db, "admin@kalam.dev")
        if existing:
            print("User already exists!")
            print(f"Email:   {existing.email}")
            print(f"API Key: {existing.api_key}")
            return

        user = await user_repo.create_user(
            db,
            email="admin@kalam.dev",
            hashed_password=None,
        )
        print("✅ User created successfully!")
        print(f"Email:   {user.email}")
        print(f"API Key: {user.api_key}")


if __name__ == "__main__":
    asyncio.run(seed())