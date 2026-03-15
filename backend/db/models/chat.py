"""
Chat Models
===========
ChatSession: A conversation thread (maps to a user + optional agent run)
ChatMessage: Individual messages within a session (user/assistant/system/tool)

Design: We store the full message content in PostgreSQL for audit/history,
and a compressed summary in Redis for fast context retrieval.
"""

import uuid
from enum import Enum as PyEnum

from sqlalchemy import (
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.dialects.postgresql import UUID, ENUM
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import BaseModel


class MessageRole(str, PyEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ChatSession(BaseModel):
    __tablename__ = "chat_sessions"

    # Foreign keys
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Session metadata
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    agent_type: Mapped[str] = mapped_column(
        String(50), default="react", nullable=False
    )  # react | multi_agent | rag_only

    # Tracking
    total_tokens_used: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    message_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Session-level configuration overrides (JSON)
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # e.g. {"model": "claude-sonnet-4-5", "temperature": 0.2, "tools": ["web_search"]}

    # Relationships
    user: Mapped["User"] = relationship(back_populates="chat_sessions")  # noqa: F821
    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
        lazy="select",
    )

    def __repr__(self) -> str:
        return f"<ChatSession id={self.id} user={self.user_id} messages={self.message_count}>"


class ChatMessage(BaseModel):
    __tablename__ = "chat_messages"

    # Foreign keys
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Message content
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    # user | assistant | system | tool
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # LLM metadata (populated for assistant messages)
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    latency_ms: Mapped[float] = mapped_column(default=0.0, nullable=False)

    # Tool call metadata (populated for tool messages)
    tool_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    tool_input: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # RAG context (which documents were retrieved for this message)
    retrieved_doc_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Relationships
    session: Mapped["ChatSession"] = relationship(back_populates="messages")

    def __repr__(self) -> str:
        return f"<ChatMessage id={self.id} role={self.role} session={self.session_id}>"