"""
Long-Term Memory (Qdrant)
=========================
Stores important facts and conversation summaries as vector embeddings.

This is what makes the agent feel like it "remembers" things across sessions.

Flow:
  End of conversation → extract key facts → embed → store in Qdrant
  Start of new conversation → embed current query → search memories → inject into context

Stored in the "long_term_memory" Qdrant collection (separate from documents).
Each memory is tagged with user_id so memories are per-user, not shared.

Memory types:
  "fact"       — specific factual statement ("User works at Acme Corp")
  "preference" — user preference ("User prefers concise responses")
  "summary"    — summary of a conversation ("Discussed RAG architecture")
  "context"    — background context ("User is building an AI agent system")
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from backend.memory.base import MemoryEntry
from backend.rag.embedder import Embedder, get_embedder
from backend.vectordb.base import DocumentChunk, VectorDB
from backend.vectordb.factory import get_vector_db
from backend.observability.logger import get_logger

logger = get_logger(__name__)


class LongTermMemory:
    """
    Semantic memory backed by Qdrant.

    Usage:
        memory = LongTermMemory()

        # Store a fact
        await memory.store(
            content="The user is building a RAG system with Python",
            user_id="user-123",
            session_id="sess-456",
            memory_type="fact",
        )

        # Recall relevant memories
        memories = await memory.search(
            query="What is the user working on?",
            user_id="user-123",
            top_k=3,
        )
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        vector_db: Optional[VectorDB] = None,
        collection: Optional[str] = None,
    ):
        self._embedder = embedder or get_embedder()
        self._vector_db = vector_db or get_vector_db()
        self._collection = collection

    def _get_collection(self) -> str:
        if self._collection:
            return self._collection
        from backend.config.settings import get_settings
        return get_settings().qdrant_collection_memory

    async def _ensure_collection(self) -> None:
        """Create the memory collection if it doesn't exist."""
        from backend.config.settings import get_settings
        collection = self._get_collection()
        if not await self._vector_db.collection_exists(collection):
            dimension = get_settings().embedding_dimensions
            await self._vector_db.create_collection(collection, dimension=dimension)

    async def store(
        self,
        content: str,
        user_id: str,
        session_id: str,
        memory_type: str = "fact",
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Embed and store a memory.

        Returns:
            The memory ID for future reference.
        """
        if not content or not content.strip():
            return ""

        await self._ensure_collection()

        memory_id = str(uuid.uuid4())
        embedding = await self._embedder.embed_one(content)

        chunk = DocumentChunk(
            id=memory_id,
            text=content,
            embedding=embedding,
            metadata={
                "user_id": user_id,
                "session_id": session_id,
                "memory_type": memory_type,
                "created_at": datetime.now(timezone.utc).isoformat(),
                **(metadata or {}),
            },
        )

        await self._vector_db.upsert(self._get_collection(), [chunk])

        logger.debug(
            "memory_stored",
            memory_id=memory_id,
            memory_type=memory_type,
            user_id=user_id,
            content_preview=content[:80],
        )

        return memory_id

    async def search(
        self,
        query: str,
        user_id: str,
        *,
        top_k: int = 3,
        score_threshold: float = 0.5,
    ) -> list[MemoryEntry]:
        """
        Find memories relevant to the query for a specific user.

        Filters by user_id so users only see their own memories.
        """
        collection = self._get_collection()

        if not await self._vector_db.collection_exists(collection):
            return []

        try:
            query_vector = await self._embedder.embed_one(query)
        except Exception as e:
            logger.error("memory_search_embedding_failed", error=str(e))
            return []

        try:
            results = await self._vector_db.search(
                collection=collection,
                query_vector=query_vector,
                top_k=top_k * 2,  # Over-fetch to allow user_id filtering
                score_threshold=score_threshold,
                filter_metadata={"user_id": user_id},
            )
        except Exception as e:
            logger.error("memory_search_failed", error=str(e))
            return []

        # Filter to this user's memories and limit to top_k
        entries = []
        for result in results:
            if result.metadata.get("user_id") != user_id:
                continue
            entries.append(MemoryEntry(
                id=result.id,
                content=result.text,
                memory_type=result.metadata.get("memory_type", "fact"),
                session_id=result.metadata.get("session_id", ""),
                user_id=user_id,
                metadata=result.metadata,
                relevance_score=result.score,
            ))
            if len(entries) >= top_k:
                break

        logger.debug(
            "memory_search_completed",
            query=query[:80],
            user_id=user_id,
            results=len(entries),
        )

        return entries

    async def store_conversation_summary(
        self,
        session_id: str,
        user_id: str,
        messages: list[dict],
    ) -> str:
        """
        Extract and store a summary of a conversation.
        Called at the end of a session to build long-term memory.

        This is a lightweight extraction — it just stores the last
        user message and assistant response as a summary.
        For production, replace this with an LLM-based summarization call.
        """
        if not messages:
            return ""

        # Simple extraction: last user + assistant turn
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        asst_msgs = [m["content"] for m in messages if m["role"] == "assistant"]

        if not user_msgs:
            return ""

        summary_parts = []
        if user_msgs:
            summary_parts.append(f"User asked: {user_msgs[-1][:200]}")
        if asst_msgs:
            summary_parts.append(f"Assistant answered: {asst_msgs[-1][:200]}")

        summary = " | ".join(summary_parts)

        return await self.store(
            content=summary,
            user_id=user_id,
            session_id=session_id,
            memory_type="summary",
            metadata={"turn_count": len(messages)},
        )

    async def delete_user_memories(self, user_id: str) -> int:
        """Delete all memories for a user (GDPR compliance)."""
        collection = self._get_collection()
        deleted = await self._vector_db.delete_by_doc_id(collection, user_id)
        logger.info("user_memories_deleted", user_id=user_id, count=deleted)
        return deleted


# Module-level singleton
_long_term_memory: Optional[LongTermMemory] = None


def get_long_term_memory() -> LongTermMemory:
    """Get the global LongTermMemory singleton."""
    global _long_term_memory
    if _long_term_memory is None:
        _long_term_memory = LongTermMemory()
    return _long_term_memory