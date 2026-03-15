"""
Qdrant Adapter
==============
Implements the VectorDB Protocol using Qdrant.

Why Qdrant?
- Open source, self-hostable (no vendor lock-in)
- Excellent Python async client
- Supports hybrid search (dense + sparse vectors)
- Payload filtering without extra indexes
- Runs in-memory for testing (no Docker required)

Configuration:
    QDRANT_URL=http://localhost:6333      # Local instance
    QDRANT_API_KEY=                       # Leave empty for local
    QDRANT_COLLECTION_DOCUMENTS=documents
    QDRANT_COLLECTION_MEMORY=long_term_memory

In-memory mode (for tests/dev without a running Qdrant):
    QdrantAdapter(in_memory=True)
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from backend.vectordb.base import (
    CollectionInfo,
    CollectionNotFoundError,
    DocumentChunk,
    SearchResult,
    VectorDBError,
)
from backend.observability.logger import get_logger

logger = get_logger(__name__)


class QdrantAdapter:
    """
    Qdrant implementation of the VectorDB Protocol.

    Usage:
        # Production (local Qdrant server)
        db = QdrantAdapter()

        # Tests / dev (no server needed)
        db = QdrantAdapter(in_memory=True)

        # Create collection and insert chunks
        await db.create_collection("documents", dimension=1536)
        await db.upsert("documents", chunks)
        results = await db.search("documents", query_vector, top_k=5)
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        in_memory: bool = False,
        timeout: int = 30,
    ):
        self._in_memory = in_memory
        self._client = None
        self._url = url
        self._api_key = api_key
        self._timeout = timeout

    async def _get_client(self):
        """Lazy-initialize the Qdrant async client."""
        if self._client is not None:
            return self._client

        try:
            from qdrant_client import AsyncQdrantClient
        except ImportError:
            raise ImportError(
                "qdrant-client not installed. Run: pip install qdrant-client"
            )

        if self._in_memory:
            # In-memory mode: perfect for tests, no server needed
            self._client = AsyncQdrantClient(":memory:")
            logger.info("qdrant_client_initialized", mode="in_memory")
        else:
            from backend.config.settings import get_settings
            settings = get_settings()
            url = self._url or settings.qdrant_url
            api_key = self._api_key or settings.qdrant_api_key

            self._client = AsyncQdrantClient(
                url=url,
                api_key=api_key or None,
                timeout=self._timeout,
            )
            logger.info("qdrant_client_initialized", mode="server", url=url)

        return self._client

    async def create_collection(
        self,
        collection: str,
        dimension: int,
    ) -> bool:
        """
        Create a Qdrant collection with cosine similarity.
        Returns True if created, False if already existed.
        """
        from qdrant_client.models import Distance, VectorParams

        client = await self._get_client()

        if await self.collection_exists(collection):
            logger.debug("collection_already_exists", collection=collection)
            return False

        await client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE,
                # on_disk=True  # Enable for large collections to save RAM
            ),
        )

        logger.info(
            "collection_created",
            collection=collection,
            dimension=dimension,
        )
        return True

    async def collection_exists(self, collection: str) -> bool:
        """Check if a collection exists."""
        try:
            client = await self._get_client()
            collections = await client.get_collections()
            names = [c.name for c in collections.collections]
            return collection in names
        except Exception:
            return False

    async def upsert(
        self,
        collection: str,
        chunks: list[DocumentChunk],
    ) -> int:
        """
        Insert or update chunks. Uses batch upsert for efficiency.
        Automatically creates collection if it doesn't exist.
        """
        if not chunks:
            return 0

        from qdrant_client.models import PointStruct

        client = await self._get_client()

        # Auto-create collection if missing
        if not await self.collection_exists(collection):
            dimension = len(chunks[0].embedding)
            await self.create_collection(collection, dimension)

        # Build Qdrant points
        points = []
        for chunk in chunks:
            # Qdrant requires UUID or integer IDs
            # We use a deterministic UUID from the chunk string ID
            point_id = self._str_to_uuid(chunk.id)

            payload = {
                "chunk_id": chunk.id,
                "text": chunk.text,
                **chunk.metadata,
            }

            points.append(PointStruct(
                id=point_id,
                vector=chunk.embedding,
                payload=payload,
            ))

        # Batch upsert (Qdrant handles this efficiently)
        await client.upsert(
            collection_name=collection,
            points=points,
            wait=True,   # Wait for indexing to complete
        )

        logger.debug(
            "chunks_upserted",
            collection=collection,
            count=len(points),
        )
        return len(points)

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
        Semantic similarity search using cosine distance.
        Optionally filter by metadata key/value pairs.
        """
        if not await self.collection_exists(collection):
            raise CollectionNotFoundError(
                f"Collection '{collection}' does not exist. "
                "Upload documents first."
            )

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = await self._get_client()

        # Build metadata filter if provided
        qdrant_filter = None
        if filter_metadata:
            conditions = [
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                )
                for key, value in filter_metadata.items()
            ]
            qdrant_filter = Filter(must=conditions)

        results = await client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold if score_threshold > 0 else None,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        search_results = []
        for hit in results:
            payload = hit.payload or {}
            text = payload.pop("text", "")
            # Remove internal field from metadata
            payload.pop("chunk_id", None)

            search_results.append(SearchResult(
                id=payload.get("chunk_id", str(hit.id)),
                text=text,
                score=hit.score,
                metadata=payload,
            ))

        logger.debug(
            "search_completed",
            collection=collection,
            results=len(search_results),
            top_k=top_k,
        )

        return search_results

    async def delete_by_doc_id(
        self,
        collection: str,
        doc_id: str,
    ) -> int:
        """Delete all chunks belonging to a document."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        if not await self.collection_exists(collection):
            return 0

        client = await self._get_client()

        # Get count before deletion for return value
        count_before = await self.count(collection)

        await client.delete(
            collection_name=collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=doc_id),
                    )
                ]
            ),
            wait=True,
        )

        count_after = await self.count(collection)
        deleted = count_before - count_after

        logger.info(
            "chunks_deleted_by_doc",
            collection=collection,
            doc_id=doc_id,
            deleted=deleted,
        )
        return deleted

    async def delete_by_ids(
        self,
        collection: str,
        ids: list[str],
    ) -> int:
        """Delete specific chunks by their string IDs."""
        if not ids or not await self.collection_exists(collection):
            return 0

        client = await self._get_client()

        uuid_ids = [self._str_to_uuid(id_) for id_ in ids]

        await client.delete(
            collection_name=collection,
            points_selector=uuid_ids,
            wait=True,
        )

        logger.debug("chunks_deleted_by_ids", collection=collection, count=len(ids))
        return len(ids)

    async def get_collection_info(
        self,
        collection: str,
    ) -> Optional[CollectionInfo]:
        """Get stats about a collection."""
        if not await self.collection_exists(collection):
            return None

        client = await self._get_client()
        info = await client.get_collection(collection)

        return CollectionInfo(
            name=collection,
            vector_count=info.vectors_count or 0,
            dimension=info.config.params.vectors.size,
            disk_size_bytes=getattr(info, "disk_data_size", 0) or 0,
            on_disk=False,
        )

    async def count(self, collection: str) -> int:
        """Count total vectors in a collection."""
        if not await self.collection_exists(collection):
            return 0

        client = await self._get_client()
        result = await client.count(collection_name=collection)
        return result.count

    async def close(self) -> None:
        """Close the client connection."""
        if self._client:
            await self._client.close()
            self._client = None

    @staticmethod
    def _str_to_uuid(s: str) -> str:
        """
        Convert a string ID to a UUID string deterministically.
        Qdrant requires UUID or integer point IDs.
        We use UUID5 (deterministic) so the same chunk ID always
        maps to the same UUID — enabling idempotent upserts.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))