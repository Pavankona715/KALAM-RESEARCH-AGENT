"""
Memory Manager
==============
Unified interface that combines short-term and long-term memory.

This is what the agent uses — it never calls ShortTermMemory or
LongTermMemory directly. The manager decides what to load, when
to save, and how to format memory for the LLM context.

Integration with the agent:
  1. Before each turn: load_context() → injects memories into agent state
  2. After each turn:  save_turn()    → persists to Redis + optionally Qdrant
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from backend.memory.base import ConversationTurn, MemoryEntry
from backend.memory.short_term import ShortTermMemory, get_short_term_memory
from backend.memory.long_term import LongTermMemory, get_long_term_memory
from backend.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryContext:
    """
    All memory context loaded for a single agent turn.
    Passed to the ContextBuilder before LLM call.
    """
    conversation_history: list[dict]      # Recent turns for LLM messages
    relevant_memories: list[str]          # Relevant long-term memory snippets
    session_turn_count: int               # How many turns in current session


class MemoryManager:
    """
    Coordinates short-term and long-term memory for agent turns.

    Usage:
        manager = MemoryManager()

        # Before the agent runs — load context
        context = await manager.load_context(
            session_id="sess-1",
            user_id="user-1",
            current_query="What is RAG?",
        )
        # context.conversation_history → inject into agent
        # context.relevant_memories   → inject into system prompt

        # After the agent responds — save the turn
        await manager.save_turn(
            session_id="sess-1",
            user_id="user-1",
            user_message="What is RAG?",
            assistant_response="RAG stands for...",
        )
    """

    def __init__(
        self,
        short_term: Optional[ShortTermMemory] = None,
        long_term: Optional[LongTermMemory] = None,
        history_limit: int = 20,
        memory_top_k: int = 3,
        enable_long_term: bool = True,
    ):
        self._short_term = short_term or get_short_term_memory()
        self._long_term = long_term or get_long_term_memory()
        self._history_limit = history_limit
        self._memory_top_k = memory_top_k
        self._enable_long_term = enable_long_term

    async def load_context(
        self,
        session_id: str,
        user_id: str,
        current_query: str,
    ) -> MemoryContext:
        """
        Load all memory context needed for an agent turn.

        Runs short-term and long-term retrieval concurrently
        for minimal latency impact.
        """
        import asyncio

        # Run both in parallel
        history_task = self._short_term.get_history_as_dicts(
            session_id, limit=self._history_limit
        )
        count_task = self._short_term.get_session_count(session_id)

        if self._enable_long_term:
            memory_task = self._long_term.search(
                query=current_query,
                user_id=user_id,
                top_k=self._memory_top_k,
                score_threshold=0.5,
            )
            history, turn_count, long_term_results = await asyncio.gather(
                history_task, count_task, memory_task,
                return_exceptions=True,
            )
        else:
            history, turn_count = await asyncio.gather(
                history_task, count_task,
                return_exceptions=True,
            )
            long_term_results = []

        # Handle exceptions gracefully — memory failure shouldn't break the agent
        if isinstance(history, Exception):
            logger.warning("short_term_load_failed", error=str(history))
            history = []

        if isinstance(turn_count, Exception):
            turn_count = 0

        if isinstance(long_term_results, Exception):
            logger.warning("long_term_search_failed", error=str(long_term_results))
            long_term_results = []

        # Format long-term memories as readable snippets
        memory_snippets = [
            f"[Past memory, relevance {m.relevance_score:.2f}]: {m.content}"
            for m in (long_term_results or [])
            if isinstance(m, MemoryEntry)
        ]

        logger.debug(
            "memory_context_loaded",
            session_id=session_id,
            history_turns=len(history),
            long_term_memories=len(memory_snippets),
        )

        return MemoryContext(
            conversation_history=history if isinstance(history, list) else [],
            relevant_memories=memory_snippets,
            session_turn_count=turn_count if isinstance(turn_count, int) else 0,
        )

    async def save_turn(
        self,
        session_id: str,
        user_id: str,
        user_message: str,
        assistant_response: str,
        save_to_long_term: bool = False,
    ) -> None:
        """
        Persist a completed conversation turn.

        Args:
            session_id: Current session ID
            user_id: User ID for memory ownership
            user_message: What the user said
            assistant_response: What the agent responded
            save_to_long_term: Whether to also save to semantic memory
                               (set True for important exchanges)
        """
        import asyncio

        # Always save to short-term — sequential to preserve order (user first)
        await self._short_term.save_turn(
            session_id=session_id,
            role="user",
            content=user_message,
        )
        await self._short_term.save_turn(
            session_id=session_id,
            role="assistant",
            content=assistant_response,
        )

        # Optionally save to long-term memory
        if save_to_long_term and self._enable_long_term:
            try:
                await self._long_term.store_conversation_summary(
                    session_id=session_id,
                    user_id=user_id,
                    messages=[
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": assistant_response},
                    ],
                )
            except Exception as e:
                logger.warning("long_term_save_failed", error=str(e))

    async def clear_session(self, session_id: str) -> None:
        """Clear short-term memory for a session."""
        await self._short_term.clear_session(session_id)

    async def store_memory(
        self,
        content: str,
        user_id: str,
        session_id: str,
        memory_type: str = "fact",
    ) -> str:
        """Explicitly store a fact in long-term memory."""
        return await self._long_term.store(
            content=content,
            user_id=user_id,
            session_id=session_id,
            memory_type=memory_type,
        )


# Module-level singleton
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get the global MemoryManager singleton."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager