"""
Database Initialization Script
================================
Creates all PostgreSQL tables from SQLAlchemy models.
Run once before starting the application for the first time.

Usage: python scripts/setup_db.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    from backend.config.settings import get_settings
    from backend.db.base import Base
    from backend.db.models import (  # noqa: F401 — import triggers model registration
        AgentRun, ChatMessage, ChatSession, Document, User,
    )

    # Import engine creation
    from sqlalchemy.ext.asyncio import create_async_engine
    settings = get_settings()

    print(f"Connecting to: {settings.database_url[:50]}...")
    engine = create_async_engine(settings.database_url, echo=True)

    async with engine.begin() as conn:
        print("Creating tables...")
        await conn.run_sync(Base.metadata.create_all)

    await engine.dispose()
    print("\n✅ Database tables created successfully!")
    print("\nTables created:")
    for table in Base.metadata.tables:
        print(f"  - {table}")


if __name__ == "__main__":
    asyncio.run(main())