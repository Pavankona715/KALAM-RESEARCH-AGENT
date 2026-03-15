"""
Agent Run Model
===============
Records every agent workflow execution for debugging, analytics, and replay.
Captures the full reasoning trace, tool calls, and final output.
"""

import uuid
from enum import Enum as PyEnum

from sqlalchemy import Float, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import BaseModel


class AgentRunStatus(str, PyEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentRun(BaseModel):
    __tablename__ = "agent_runs"

    # Links
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Agent metadata
    agent_type: Mapped[str] = mapped_column(String(50), nullable=False)
    # react | planner | researcher | analyst | writer | multi_agent

    # Input / Output
    input_query: Mapped[str] = mapped_column(Text, nullable=False)
    final_output: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Execution status
    status: Mapped[str] = mapped_column(
        String(20), default=AgentRunStatus.RUNNING, nullable=False, index=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Performance metrics
    total_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    steps_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Full reasoning trace (LangGraph state snapshots)
    # Stored as JSON array of step objects:
    # [{"step": 1, "type": "thought|action|observation", "content": "..."}]
    reasoning_trace: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Tools used during this run
    tools_used: Mapped[list | None] = mapped_column(JSON, nullable=True)
    # ["web_search", "calculator"]

    # LangSmith run ID for cross-referencing traces
    langsmith_run_id: Mapped[str | None] = mapped_column(String(100), nullable=True)

    def __repr__(self) -> str:
        return (
            f"<AgentRun id={self.id} "
            f"type={self.agent_type} "
            f"status={self.status}>"
        )