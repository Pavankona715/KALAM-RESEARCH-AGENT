"""
Short-Term Memory (Redis)
=========================
Stores recent conversation history in Redis.

Design:
- Each session gets a Redis list key: "memory:session:{session_id}"
- Messages are stored as JSON, newest appended to the right
- TTL resets on every write (keeps active sessions alive)
- When TTL expires, session data is deleted automatically

Why Redis over PostgreSQL for this?
- Sub-millisecond reads on every chat turn
- Built-in TTL — no cron job needed to purge old sessions
- List data structure maps perfectly to conversation history
- Survives process restarts (unlike in-memory dict)

Fallback: If Redis is unavailable, silently falls back to
an in-memory dict. Agent still works, just loses persistence.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from backend.memory.base import ConversationTurn
from backend.observability.logger import get_logger

logger = get_logger(__name__)

MEMORY_KEY_PREFIX = "memory:session:"
DEFAULT_TTL_SECONDS = 86400  # 24 hours


class ShortTermMemory:
    """
    Redis-backed conversation history buffer.

    Usage:
        memory = ShortTermMemory()
        await memory.save_turn("sess-1", "user", "What is RAG?")
        await memory.save_turn("sess-1", "assistant", "RAG stands for...")
        history = await memory.get_history("sess-1")
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_turns_per_session: int = 100,
    ):
        self._redis_url = redis_url
        self._ttl = ttl_seconds
        self._max_turns = max_turns_per_session
        self._client = None
        self._fallback: dict[str, list] = {}  # in-memory fallback

    async def _get_client(self):
        """Lazy-initialize Redis client."""
        if self._client is not None:
            return self._client

        try:
            import redis.asyncio as aioredis
            from backend.config.settings import get_settings

            url = self._redis_url or get_settings().redis_url
            self._client = aioredis.from_url(url, decode_responses=True)
            await self._client.ping()
            logger.info("short_term_memory_redis_connected")
        except Exception as e:
            logger.warning(
                "short_term_memory_redis_unavailable",
                error=str(e),
                note="falling back to in-memory storage",
            )
            self._client = None

        return self._client

    def _session_key(self, session_id: str) -> str:
        return f"{MEMORY_KEY_PREFIX}{session_id}"

    async def save_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        **metadata,
    ) -> None:
        """Append a conversation turn to the session history."""
        turn = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }
        turn_json = json.dumps(turn)

        client = await self._get_client()

        if client:
            try:
                key = self._session_key(session_id)
                pipe = client.pipeline()
                pipe.rpush(key, turn_json)
                # Trim to max turns to prevent unbounded growth
                pipe.ltrim(key, -self._max_turns, -1)
                # Reset TTL on every write
                pipe.expire(key, self._ttl)
                await pipe.execute()
                return
            except Exception as e:
                logger.warning("redis_write_failed", error=str(e))

        # Fallback to in-memory
        if session_id not in self._fallback:
            self._fallback[session_id] = []
        self._fallback[session_id].append(turn)
        # Trim fallback too
        if len(self._fallback[session_id]) > self._max_turns:
            self._fallback[session_id] = self._fallback[session_id][-self._max_turns:]

    async def get_history(
        self,
        session_id: str,
        limit: int = 20,
    ) -> list[ConversationTurn]:
        """
        Get recent conversation history for a session.
        Returns turns in chronological order (oldest first).
        """
        client = await self._get_client()

        if client:
            try:
                key = self._session_key(session_id)
                # Get last `limit` items from the list
                raw_turns = await client.lrange(key, -limit, -1)
                return [
                    self._deserialize_turn(t)
                    for t in raw_turns
                    if t
                ]
            except Exception as e:
                logger.warning("redis_read_failed", error=str(e))

        # Fallback
        turns = self._fallback.get(session_id, [])
        return [self._deserialize_turn(json.dumps(t)) for t in turns[-limit:]]

    async def get_history_as_dicts(
        self,
        session_id: str,
        limit: int = 20,
    ) -> list[dict]:
        """
        Get history formatted as dicts for LLM context.
        Returns: [{"role": "user", "content": "..."}, ...]
        """
        turns = await self.get_history(session_id, limit=limit)
        return [{"role": t.role, "content": t.content} for t in turns]

    async def clear_session(self, session_id: str) -> None:
        """Delete all messages for a session."""
        client = await self._get_client()

        if client:
            try:
                await client.delete(self._session_key(session_id))
                return
            except Exception as e:
                logger.warning("redis_delete_failed", error=str(e))

        self._fallback.pop(session_id, None)

    async def session_exists(self, session_id: str) -> bool:
        """Check if a session has any stored messages."""
        client = await self._get_client()

        if client:
            try:
                return await client.exists(self._session_key(session_id)) > 0
            except Exception:
                pass

        return session_id in self._fallback

    async def get_session_count(self, session_id: str) -> int:
        """Get number of turns stored for a session."""
        client = await self._get_client()

        if client:
            try:
                return await client.llen(self._session_key(session_id))
            except Exception:
                pass

        return len(self._fallback.get(session_id, []))

    @staticmethod
    def _deserialize_turn(raw: str) -> ConversationTurn:
        """Parse a JSON string into a ConversationTurn."""
        try:
            data = json.loads(raw)
            return ConversationTurn(
                role=data.get("role", "user"),
                content=data.get("content", ""),
                session_id=data.get("session_id", ""),
                timestamp=data.get("timestamp"),
                metadata={
                    k: v for k, v in data.items()
                    if k not in ("role", "content", "session_id", "timestamp", "id")
                },
            )
        except (json.JSONDecodeError, KeyError):
            return ConversationTurn(role="user", content=raw, session_id="")


# Module-level singleton
_short_term_memory: Optional[ShortTermMemory] = None


def get_short_term_memory() -> ShortTermMemory:
    """Get the global ShortTermMemory singleton."""
    global _short_term_memory
    if _short_term_memory is None:
        _short_term_memory = ShortTermMemory()
    return _short_term_memory