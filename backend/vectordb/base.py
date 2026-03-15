"""
Vector Database Interface
=========================
Defines the contract every vector DB adapter must fulfill.

Uses Python Protocol (structural typing) — no inheritance required.
Any class with the correct methods satisfies the protocol.

Data flow:
    DocumentChunk (with embedding) → VectorDB.upsert()
    SearchQuery (with vector)      → VectorDB.search() → SearchResult[]

Design decisions:
- Chunks carry their own embeddings — the VectorDB never calls the
  embedding model. That's the embedder's job (separation of concerns).
- Metadata is a free-form dict — the adapter stores it as-is.
  The RAG layer decides what metadata to attach.
- All operations are async — vector DBs are I/O bound.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


# ─── Data Types ───────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """
    A single chunk of text with its embedding vector and metadata.
    This is the unit of storage in the vector database.
    """
    id: str                           # Unique chunk ID (e.g., "doc123_chunk_0")
    text: str                         # The actual text content
    embedding: list[float]            # Dense vector from embedding model
    metadata: dict[str, Any] = field(default_factory=dict)
    # Common metadata keys:
    # - doc_id: str          parent document ID
    # - filename: str        original filename
    # - chunk_index: int     position within document
    # - source_url: str      URL if from web
    # - doc_type: str        pdf, docx, txt, url


@dataclass
class SearchResult:
    """
    A single result from a vector similarity search.
    """
    id: str                           # Chunk ID
    text: str                         # Chunk text content
    score: float                      # Similarity score (0.0 to 1.0)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def doc_id(self) -> Optional[str]:
        return self.metadata.get("doc_id")

    @property
    def filename(self) -> Optional[str]:
        return self.metadata.get("filename")


@dataclass
class CollectionInfo:
    """Stats about a vector collection."""
    name: str
    vector_count: int
    dimension: int
    disk_size_bytes: int = 0
    on_disk: bool = False


# ─── Protocol ─────────────────────────────────────────────────────────────────

@runtime_checkable
class VectorDB(Protocol):
    """
    Contract every vector DB adapter must fulfill.

    @runtime_checkable enables isinstance() checks:
        assert isinstance(my_db, VectorDB)
    """

    async def upsert(
        self,
        collection: str,
        chunks: list[DocumentChunk],
    ) -> int:
        """
        Insert or update chunks in a collection.

        Returns:
            Number of chunks successfully upserted.
        """
        ...

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        *,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """
        Find the most similar chunks to a query vector.

        Args:
            collection: Collection name to search
            query_vector: Dense embedding of the query
            top_k: Maximum results to return
            score_threshold: Minimum similarity score (0.0-1.0)
            filter_metadata: Optional key/value filter on metadata

        Returns:
            List of SearchResult ordered by score descending.
        """
        ...

    async def delete_by_doc_id(
        self,
        collection: str,
        doc_id: str,
    ) -> int:
        """
        Delete all chunks belonging to a document.

        Returns:
            Number of chunks deleted.
        """
        ...

    async def delete_by_ids(
        self,
        collection: str,
        ids: list[str],
    ) -> int:
        """
        Delete specific chunks by their IDs.

        Returns:
            Number of chunks deleted.
        """
        ...

    async def get_collection_info(
        self,
        collection: str,
    ) -> Optional[CollectionInfo]:
        """
        Get stats about a collection. Returns None if not found.
        """
        ...

    async def create_collection(
        self,
        collection: str,
        dimension: int,
    ) -> bool:
        """
        Create a collection if it doesn't exist.

        Returns:
            True if created, False if already existed.
        """
        ...

    async def collection_exists(self, collection: str) -> bool:
        """Check if a collection exists."""
        ...

    async def count(self, collection: str) -> int:
        """Count total vectors in a collection."""
        ...


# ─── Exceptions ───────────────────────────────────────────────────────────────

class VectorDBError(Exception):
    """Base exception for vector database errors."""
    pass


class CollectionNotFoundError(VectorDBError):
    """Collection does not exist."""
    pass


class VectorDimensionMismatchError(VectorDBError):
    """Vector dimension doesn't match collection configuration."""
    pass