"""
Chat Repository
===============
All database operations for ChatSession and ChatMessage models.
"""

import uuid
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.db.models.chat import ChatMessage, ChatSession, MessageRole
from backend.db.repositories.base_repo import BaseRepository


class ChatSessionRepository(BaseRepository[ChatSession]):
    model = ChatSession

    async def get_sessions_for_user(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        *,
        skip: int = 0,
        limit: int = 20,
    ) -> list[ChatSession]:
        """Get all sessions for a user, ordered by most recent."""
        result = await db.execute(
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_with_messages(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
        *,
        message_limit: int = 50,
    ) -> ChatSession | None:
        """Load a session with its recent messages (eager load)."""
        result = await db.execute(
            select(ChatSession)
            .options(selectinload(ChatSession.messages))
            .where(ChatSession.id == session_id)
        )
        session = result.scalar_one_or_none()

        # Trim to message_limit (most recent)
        if session and len(session.messages) > message_limit:
            session.messages = session.messages[-message_limit:]

        return session

    async def increment_counters(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
        *,
        tokens: int = 0,
        messages: int = 1,
    ) -> None:
        """Atomically increment token and message counters."""
        session = await self.get_by_id(db, session_id)
        if session:
            await db.execute(
                update(ChatSession)
                .where(ChatSession.id == session_id)
                .values(
                    total_tokens_used=ChatSession.total_tokens_used + tokens,
                    message_count=ChatSession.message_count + messages,
                )
            )


class ChatMessageRepository(BaseRepository[ChatMessage]):
    model = ChatMessage

    async def get_messages_for_session(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
        *,
        limit: int = 50,
        skip: int = 0,
    ) -> list[ChatMessage]:
        """Get messages for a session in chronological order."""
        result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def create_message(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
        role: MessageRole,
        content: str,
        **kwargs: Any,
    ) -> ChatMessage:
        """Create a new chat message."""
        return await self.create(
            db,
            session_id=session_id,
            role=role.value,
            content=content,
            **kwargs,
        )

    async def get_conversation_history(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
        *,
        limit: int = 20,
    ) -> list[dict]:
        """
        Returns conversation history formatted for LLM context.
        Output: [{"role": "user", "content": "..."}, ...]
        """
        messages = await self.get_messages_for_session(
            db, session_id, limit=limit
        )
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
            if msg.role in ("user", "assistant")
        ]


# Module-level singletons
chat_session_repo = ChatSessionRepository()
chat_message_repo = ChatMessageRepository()