"""
Database Session Management
============================
Creates and manages the async SQLAlchemy engine and session factory.

Design decisions:
- async engine (asyncpg driver) - never block the event loop
- Session-per-request pattern via FastAPI dependency injection
- Connection pooling configured from settings
- Explicit commit/rollback - no autocommit surprises
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.config.settings import get_settings
from backend.observability.logger import get_logger

logger = get_logger(__name__)

# Module-level engine and factory — created once at startup
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def create_engine() -> AsyncEngine:
    """Create the async SQLAlchemy engine with connection pooling."""
    settings = get_settings()

    engine = create_async_engine(
        settings.database_url,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        # Echo SQL in development for debugging
        echo=settings.app_debug and settings.is_development,
        # Pool configuration
        pool_pre_ping=True,      # Verify connections before use
        pool_recycle=3600,       # Recycle connections after 1 hour
    )

    logger.info(
        "database_engine_created",
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
    )

    return engine


def get_engine() -> AsyncEngine:
    """Get the global engine instance. Must call init_db() first."""
    if _engine is None:
        raise RuntimeError(
            "Database engine not initialized. "
            "Call init_db() during application startup."
        )
    return _engine


async def init_db() -> None:
    """
    Initialize database engine and session factory.
    Call during FastAPI lifespan startup.
    """
    global _engine, _session_factory

    _engine = create_engine()
    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,   # Don't expire objects after commit
        autocommit=False,
        autoflush=False,
    )

    # Verify connectivity
    async with _engine.connect() as conn:
        await conn.execute(__import__("sqlalchemy").text("SELECT 1"))

    logger.info("database_initialized")


async def close_db() -> None:
    """
    Close database connections.
    Call during FastAPI lifespan shutdown.
    """
    global _engine, _session_factory

    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("database_connections_closed")


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a database session per request.

    Usage in routes:
        @router.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db_session)):
            ...

    Automatically commits on success, rolls back on exception.
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialized.")

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager version for use outside of FastAPI routes
    (e.g., background tasks, scripts).

    Usage:
        async with get_db_context() as db:
            user = await user_repo.get_by_id(db, user_id)
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialized.")

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()