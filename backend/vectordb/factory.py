"""
VectorDB Factory
================
Creates the correct VectorDB implementation based on configuration.

This is the only place outside of adapters that knows which
implementation to use. Everything else depends on the VectorDB Protocol.

Usage:
    # Production
    db = get_vector_db()

    # Tests (in-memory, no server needed)
    db = get_vector_db(in_memory=True)

    # Explicitly use a specific adapter
    db = create_vector_db("qdrant", in_memory=True)
"""

from __future__ import annotations

from typing import Literal, Optional

from backend.vectordb.base import VectorDB
from backend.observability.logger import get_logger

logger = get_logger(__name__)

VectorDBBackend = Literal["qdrant"]

# Module-level singleton
_vector_db: Optional[VectorDB] = None


def create_vector_db(
    backend: VectorDBBackend = "qdrant",
    in_memory: bool = False,
) -> VectorDB:
    """
    Create a new VectorDB instance.

    Args:
        backend: Which vector DB to use. Currently only "qdrant".
        in_memory: Use in-memory mode (no server, great for tests).

    Returns:
        A VectorDB-compliant instance.
    """
    if backend == "qdrant":
        from backend.vectordb.qdrant_adapter import QdrantAdapter
        return QdrantAdapter(in_memory=in_memory)

    raise ValueError(
        f"Unknown vector DB backend: '{backend}'. "
        "Available: 'qdrant'"
    )


def get_vector_db(in_memory: bool = False) -> VectorDB:
    """
    Get the global VectorDB singleton.

    Uses the backend configured in settings (currently always Qdrant).
    Pass in_memory=True to bypass the server — useful in tests.

    Usage:
        db = get_vector_db()
        results = await db.search("documents", query_vector)
    """
    global _vector_db

    # Always create a new instance for in-memory (test isolation)
    if in_memory:
        return create_vector_db("qdrant", in_memory=True)

    if _vector_db is None:
        _vector_db = create_vector_db("qdrant", in_memory=False)
        logger.info("vector_db_initialized", backend="qdrant")

    return _vector_db


async def ensure_collections_exist(
    vector_db: Optional[VectorDB] = None,
) -> None:
    """
    Ensure all required collections exist at startup.
    Call this during application lifespan startup.

    Creates 'documents' and 'long_term_memory' collections
    if they don't already exist.
    """
    from backend.config.settings import get_settings
    settings = get_settings()

    db = vector_db or get_vector_db()

    dimension = settings.embedding_dimensions

    collections = [
        settings.qdrant_collection_documents,
        settings.qdrant_collection_memory,
    ]

    for collection in collections:
        created = await db.create_collection(collection, dimension=dimension)
        if created:
            logger.info("collection_created", collection=collection, dimension=dimension)
        else:
            logger.debug("collection_exists", collection=collection)