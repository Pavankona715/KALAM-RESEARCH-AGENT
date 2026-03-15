"""
Agent Run Repository
====================
Tracks agent workflow executions for debugging and analytics.
"""

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models.agent_run import AgentRun, AgentRunStatus
from backend.db.repositories.base_repo import BaseRepository


class AgentRunRepository(BaseRepository[AgentRun]):
    model = AgentRun

    async def get_runs_for_user(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        *,
        skip: int = 0,
        limit: int = 20,
        status: AgentRunStatus | None = None,
    ) -> list[AgentRun]:
        """Get agent runs for a user with optional status filter."""
        query = (
            select(AgentRun)
            .where(AgentRun.user_id == user_id)
            .order_by(AgentRun.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        if status:
            query = query.where(AgentRun.status == status.value)

        result = await db.execute(query)
        return list(result.scalars().all())

    async def start_run(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        agent_type: str,
        input_query: str,
        session_id: uuid.UUID | None = None,
    ) -> AgentRun:
        """Create a new agent run record in RUNNING state."""
        return await self.create(
            db,
            user_id=user_id,
            session_id=session_id,
            agent_type=agent_type,
            input_query=input_query,
            status=AgentRunStatus.RUNNING.value,
        )

    async def complete_run(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        *,
        output: str,
        tokens: int,
        latency_ms: float,
        steps: int,
        reasoning_trace: list | None = None,
        tools_used: list | None = None,
        langsmith_run_id: str | None = None,
    ) -> AgentRun | None:
        """Mark a run as completed with results."""
        return await self.update(
            db,
            run_id,
            status=AgentRunStatus.COMPLETED.value,
            final_output=output,
            total_tokens=tokens,
            latency_ms=latency_ms,
            steps_count=steps,
            reasoning_trace=reasoning_trace,
            tools_used=tools_used,
            langsmith_run_id=langsmith_run_id,
        )

    async def fail_run(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        error: str,
    ) -> AgentRun | None:
        """Mark a run as failed with error message."""
        return await self.update(
            db,
            run_id,
            status=AgentRunStatus.FAILED.value,
            error_message=error,
        )


# Module-level singleton
agent_run_repo = AgentRunRepository()