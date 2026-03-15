"""
Memory System Unit Tests
========================
Tests for ShortTermMemory (Redis), LongTermMemory (Qdrant), and MemoryManager.

- ShortTermMemory: tests use the in-memory fallback (no Redis server needed)
- LongTermMemory: tests use in-memory Qdrant + mock embedder
- MemoryManager: tests mock both backends
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.memory.base import ConversationTurn, MemoryEntry


# ─── ShortTermMemory Tests ────────────────────────────────────────────────────

class TestShortTermMemory:
    """
    Tests use the in-memory fallback — no Redis server needed.
    The fallback activates automatically when Redis is unreachable.
    """

    def setup_method(self):
        from backend.memory.short_term import ShortTermMemory
        # Use an invalid URL to force the in-memory fallback
        self.memory = ShortTermMemory(
            redis_url="redis://localhost:1",  # Unreachable → fallback
            ttl_seconds=3600,
        )

    @pytest.mark.asyncio
    async def test_save_and_get_history(self):
        await self.memory.save_turn("sess-1", "user", "Hello")
        await self.memory.save_turn("sess-1", "assistant", "Hi there!")

        history = await self.memory.get_history("sess-1")
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "Hello"
        assert history[1].role == "assistant"
        assert history[1].content == "Hi there!"

    @pytest.mark.asyncio
    async def test_get_history_empty_session(self):
        history = await self.memory.get_history("nonexistent-session")
        assert history == []

    @pytest.mark.asyncio
    async def test_limit_respected(self):
        for i in range(10):
            await self.memory.save_turn("sess-limit", "user", f"Message {i}")

        history = await self.memory.get_history("sess-limit", limit=3)
        assert len(history) == 3
        # Should be the most recent 3
        assert history[-1].content == "Message 9"

    @pytest.mark.asyncio
    async def test_get_history_as_dicts(self):
        await self.memory.save_turn("sess-2", "user", "Question")
        await self.memory.save_turn("sess-2", "assistant", "Answer")

        dicts = await self.memory.get_history_as_dicts("sess-2")
        assert dicts == [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]

    @pytest.mark.asyncio
    async def test_session_exists_true(self):
        await self.memory.save_turn("sess-exists", "user", "Hello")
        assert await self.memory.session_exists("sess-exists") is True

    @pytest.mark.asyncio
    async def test_session_exists_false(self):
        assert await self.memory.session_exists("sess-ghost") is False

    @pytest.mark.asyncio
    async def test_clear_session(self):
        await self.memory.save_turn("sess-clear", "user", "Hello")
        assert await self.memory.session_exists("sess-clear") is True
        await self.memory.clear_session("sess-clear")
        assert await self.memory.session_exists("sess-clear") is False

    @pytest.mark.asyncio
    async def test_get_session_count(self):
        for i in range(5):
            await self.memory.save_turn("sess-count", "user", f"Msg {i}")
        count = await self.memory.get_session_count("sess-count")
        assert count == 5

    @pytest.mark.asyncio
    async def test_multiple_sessions_isolated(self):
        """Messages in one session don't appear in another."""
        await self.memory.save_turn("sess-a", "user", "Message in A")
        await self.memory.save_turn("sess-b", "user", "Message in B")

        history_a = await self.memory.get_history("sess-a")
        history_b = await self.memory.get_history("sess-b")

        assert len(history_a) == 1
        assert history_a[0].content == "Message in A"
        assert len(history_b) == 1
        assert history_b[0].content == "Message in B"

    @pytest.mark.asyncio
    async def test_max_turns_cap(self):
        """Memory is trimmed to max_turns_per_session."""
        memory = __import__(
            'backend.memory.short_term', fromlist=['ShortTermMemory']
        ).ShortTermMemory(redis_url="redis://localhost:1", max_turns_per_session=5)

        for i in range(10):
            await memory.save_turn("sess-max", "user", f"Msg {i}")

        history = await memory.get_history("sess-max", limit=100)
        assert len(history) <= 5

    def test_deserialize_turn_valid_json(self):
        import json
        from backend.memory.short_term import ShortTermMemory
        raw = json.dumps({"role": "user", "content": "hello", "session_id": "s1"})
        turn = ShortTermMemory._deserialize_turn(raw)
        assert turn.role == "user"
        assert turn.content == "hello"

    def test_deserialize_turn_invalid_json(self):
        from backend.memory.short_term import ShortTermMemory
        turn = ShortTermMemory._deserialize_turn("not json")
        assert turn.content == "not json"


# ─── LongTermMemory Tests ─────────────────────────────────────────────────────

class TestLongTermMemory:
    @pytest.fixture
    async def long_term_memory(self):
        """Async fixture: LongTermMemory with in-memory Qdrant + mock embedder."""
        from backend.memory.long_term import LongTermMemory
        from backend.vectordb.qdrant_adapter import QdrantAdapter
        from backend.rag.embedder import Embedder

        dim = 4
        mock_embedder = MagicMock(spec=Embedder)
        mock_embedder.embed_one = AsyncMock(return_value=[0.5] * dim)
        mock_embedder.embed_many = AsyncMock(return_value=[[0.5] * dim])

        db = QdrantAdapter(in_memory=True)
        await db.create_collection("test_memory", dimension=dim)

        memory = LongTermMemory(
            embedder=mock_embedder,
            vector_db=db,
            collection="test_memory",
        )
        yield memory
        await db.close()

    @pytest.mark.asyncio
    async def test_store_returns_id(self, long_term_memory):
        memory_id = await long_term_memory.store(
            content="User is building an AI agent",
            user_id="user-1",
            session_id="sess-1",
        )
        assert memory_id != ""
        assert len(memory_id) > 0

    @pytest.mark.asyncio
    async def test_store_empty_content_returns_empty(self, long_term_memory):
        result = await long_term_memory.store("", user_id="user-1", session_id="sess-1")
        assert result == ""

    @pytest.mark.asyncio
    async def test_store_and_search(self, long_term_memory):
        await long_term_memory.store(
            content="User works at Acme Corp as a senior engineer",
            user_id="user-1",
            session_id="sess-1",
            memory_type="fact",
        )
        results = await long_term_memory.search(
            query="Where does the user work?",
            user_id="user-1",
        )
        assert len(results) > 0
        assert all(isinstance(r, MemoryEntry) for r in results)
        assert all(r.user_id == "user-1" for r in results)

    @pytest.mark.asyncio
    async def test_search_empty_collection_returns_empty(self, long_term_memory):
        results = await long_term_memory.search(query="anything", user_id="user-1")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_returns_memory_entries(self, long_term_memory):
        await long_term_memory.store("User prefers Python", "user-1", "sess-1", "preference")
        results = await long_term_memory.search("programming language", "user-1")
        for result in results:
            assert hasattr(result, "content")
            assert hasattr(result, "memory_type")
            assert hasattr(result, "relevance_score")

    @pytest.mark.asyncio
    async def test_store_conversation_summary(self, long_term_memory):
        messages = [
            {"role": "user", "content": "What is RAG?"},
            {"role": "assistant", "content": "RAG stands for Retrieval Augmented Generation"},
        ]
        memory_id = await long_term_memory.store_conversation_summary(
            session_id="sess-1",
            user_id="user-1",
            messages=messages,
        )
        assert memory_id != ""

    @pytest.mark.asyncio
    async def test_store_conversation_summary_empty_messages(self, long_term_memory):
        result = await long_term_memory.store_conversation_summary("sess-1", "user-1", [])
        assert result == ""

    @pytest.mark.asyncio
    async def test_embedding_failure_returns_empty(self):
        from backend.memory.long_term import LongTermMemory
        from backend.vectordb.qdrant_adapter import QdrantAdapter
        from backend.rag.embedder import Embedder

        mock_embedder = MagicMock(spec=Embedder)
        mock_embedder.embed_one = AsyncMock(side_effect=RuntimeError("API down"))

        memory = LongTermMemory(
            embedder=mock_embedder,
            vector_db=QdrantAdapter(in_memory=True),
            collection="test_memory",
        )
        results = await memory.search("query", "user-1")
        assert results == []


# ─── MemoryManager Tests ──────────────────────────────────────────────────────

class TestMemoryManager:
    def _make_manager(self):
        from backend.memory.manager import MemoryManager
        from backend.memory.short_term import ShortTermMemory
        from backend.memory.long_term import LongTermMemory

        short_term = ShortTermMemory(redis_url="redis://localhost:1")
        mock_long_term = MagicMock(spec=LongTermMemory)
        mock_long_term.search = AsyncMock(return_value=[])
        mock_long_term.store_conversation_summary = AsyncMock(return_value="mem-id")

        manager = MemoryManager(
            short_term=short_term,
            long_term=mock_long_term,
            enable_long_term=True,
        )
        return manager, mock_long_term

    @pytest.mark.asyncio
    async def test_load_context_returns_memory_context(self):
        from backend.memory.manager import MemoryContext
        manager, _ = self._make_manager()

        context = await manager.load_context(
            session_id="sess-1",
            user_id="user-1",
            current_query="What is Python?",
        )

        assert isinstance(context, MemoryContext)
        assert isinstance(context.conversation_history, list)
        assert isinstance(context.relevant_memories, list)
        assert isinstance(context.session_turn_count, int)

    @pytest.mark.asyncio
    async def test_load_context_includes_history(self):
        manager, _ = self._make_manager()

        # Save some turns first
        await manager._short_term.save_turn("sess-2", "user", "Hello")
        await manager._short_term.save_turn("sess-2", "assistant", "Hi!")

        context = await manager.load_context("sess-2", "user-1", "Follow up")
        assert len(context.conversation_history) == 2

    @pytest.mark.asyncio
    async def test_save_turn_persists_to_short_term(self):
        manager, _ = self._make_manager()

        await manager.save_turn(
            session_id="sess-save",
            user_id="user-1",
            user_message="What is RAG?",
            assistant_response="RAG is Retrieval Augmented Generation.",
        )

        history = await manager._short_term.get_history("sess-save")
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_save_turn_with_long_term(self):
        manager, mock_long_term = self._make_manager()

        await manager.save_turn(
            session_id="sess-lt",
            user_id="user-1",
            user_message="Remember: I prefer Python",
            assistant_response="Got it!",
            save_to_long_term=True,
        )

        mock_long_term.store_conversation_summary.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_turn_without_long_term(self):
        manager, mock_long_term = self._make_manager()

        await manager.save_turn(
            session_id="sess-no-lt",
            user_id="user-1",
            user_message="Hello",
            assistant_response="Hi",
            save_to_long_term=False,
        )

        mock_long_term.store_conversation_summary.assert_not_called()

    @pytest.mark.asyncio
    async def test_clear_session(self):
        manager, _ = self._make_manager()

        await manager._short_term.save_turn("sess-clear", "user", "Hello")
        assert await manager._short_term.session_exists("sess-clear")

        await manager.clear_session("sess-clear")
        assert not await manager._short_term.session_exists("sess-clear")

    @pytest.mark.asyncio
    async def test_long_term_failure_does_not_break_context_load(self):
        """Memory manager is resilient — failures don't crash the agent."""
        from backend.memory.manager import MemoryManager
        from backend.memory.short_term import ShortTermMemory
        from backend.memory.long_term import LongTermMemory

        short_term = ShortTermMemory(redis_url="redis://localhost:1")
        mock_long_term = MagicMock(spec=LongTermMemory)
        mock_long_term.search = AsyncMock(side_effect=RuntimeError("Qdrant down"))

        manager = MemoryManager(
            short_term=short_term,
            long_term=mock_long_term,
            enable_long_term=True,
        )

        # Should not raise even with long-term failure
        context = await manager.load_context("sess-1", "user-1", "query")
        assert context.relevant_memories == []


# ─── MemoryEntry + ConversationTurn Tests ─────────────────────────────────────

class TestMemoryDataTypes:
    def test_memory_entry_creation(self):
        entry = MemoryEntry(
            id="mem-1",
            content="User works at Acme",
            memory_type="fact",
            session_id="sess-1",
            user_id="user-1",
        )
        assert entry.id == "mem-1"
        assert entry.memory_type == "fact"
        assert entry.embedding is None
        assert entry.relevance_score is None

    def test_conversation_turn_creation(self):
        turn = ConversationTurn(
            role="user",
            content="What is RAG?",
            session_id="sess-1",
        )
        assert turn.role == "user"
        assert turn.content == "What is RAG?"
        assert turn.metadata == {}