"""
Memory Interface
================
Defines the contract for all memory implementations.

Two memory types:
- ShortTermMemory: Fast key-value store, TTL-based, recent messages
- LongTermMemory:  Vector store, semantic search, persistent facts

Both implement the same Protocol so they're interchangeable in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class MemoryEntry:
    """
    A single unit of stored memory.

    Used for both short-term (conversation messages) and
    long-term (semantic facts extracted from conversations).
    """
    id: str
    content: str                          # The actual text content
    memory_type: str                      # "conversation" | "fact" | "summary"
    session_id: str
    user_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None

    # Only populated for long-term memory entries
    embedding: Optional[list[float]] = None
    relevance_score: Optional[float] = None


@dataclass
class ConversationTurn:
    """A single exchange in a conversation."""
    role: str          # "user" | "assistant"
    content: str
    session_id: str
    timestamp: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ShortTermMemoryProtocol(Protocol):
    """Contract for short-term (conversation) memory."""

    async def save_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        **metadata,
    ) -> None:
        """Save a single conversation turn."""
        ...

    async def get_history(
        self,
        session_id: str,
        limit: int = 20,
    ) -> list[ConversationTurn]:
        """Retrieve recent conversation history."""
        ...

    async def clear_session(self, session_id: str) -> None:
        """Delete all messages for a session."""
        ...

    async def session_exists(self, session_id: str) -> bool:
        """Check if a session has any stored messages."""
        ...


@runtime_checkable
class LongTermMemoryProtocol(Protocol):
    """Contract for long-term (semantic) memory."""

    async def store(
        self,
        content: str,
        user_id: str,
        session_id: str,
        memory_type: str = "fact",
        metadata: Optional[dict] = None,
    ) -> str:
        """Store a memory. Returns the memory ID."""
        ...

    async def search(
        self,
        query: str,
        user_id: str,
        *,
        top_k: int = 3,
        score_threshold: float = 0.5,
    ) -> list[MemoryEntry]:
        """Search for relevant memories by semantic similarity."""
        ...

    async def delete_user_memories(self, user_id: str) -> int:
        """Delete all memories for a user. Returns count deleted."""
        ...