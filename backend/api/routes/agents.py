"""
Agents Endpoint
===============
GET  /agents/runs — List agent execution history.
GET  /agents/runs/{run_id} — Get full details of a specific run (reasoning trace).
POST /agents/runs/{run_id}/cancel — Cancel a running agent.
"""

import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from backend.api.dependencies import CurrentUser, DBSession, Pagination
from backend.db.models.agent_run import AgentRunStatus
from backend.db.repositories.agent_run_repo import agent_run_repo
from backend.observability.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/agents", tags=["Agents"])


class AgentRunSummary(BaseModel):
    id: uuid.UUID
    agent_type: str
    status: str
    input_query: str
    total_tokens: int
    latency_ms: float
    steps_count: int
    created_at: str


class AgentRunDetail(AgentRunSummary):
    final_output: Optional[str]
    reasoning_trace: Optional[list]
    tools_used: Optional[list]
    error_message: Optional[str]
    langsmith_run_id: Optional[str]


@router.get("/runs", response_model=list[AgentRunSummary])
async def list_agent_runs(
    db: DBSession,
    user: CurrentUser,
    pagination: Pagination,
    status_filter: Optional[str] = None,
) -> list[AgentRunSummary]:
    """List recent agent execution runs for the current user."""
    status_enum = None
    if status_filter:
        try:
            status_enum = AgentRunStatus(status_filter)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid status. Valid values: {[s.value for s in AgentRunStatus]}",
            )

    runs = await agent_run_repo.get_runs_for_user(
        db,
        user.id,
        skip=pagination.skip,
        limit=pagination.limit,
        status=status_enum,
    )
    return [
        AgentRunSummary(
            id=r.id,
            agent_type=r.agent_type,
            status=r.status,
            input_query=r.input_query[:200],
            total_tokens=r.total_tokens,
            latency_ms=r.latency_ms,
            steps_count=r.steps_count,
            created_at=r.created_at.isoformat(),
        )
        for r in runs
    ]


@router.get("/runs/{run_id}", response_model=AgentRunDetail)
async def get_agent_run(
    run_id: uuid.UUID,
    db: DBSession,
    user: CurrentUser,
) -> AgentRunDetail:
    """Get full details of an agent run including reasoning trace."""
    run = await agent_run_repo.get_by_id(db, run_id)

    if not run or run.user_id != user.id:
        raise HTTPException(status_code=404, detail=f"Agent run {run_id} not found")

    return AgentRunDetail(
        id=run.id,
        agent_type=run.agent_type,
        status=run.status,
        input_query=run.input_query,
        final_output=run.final_output,
        total_tokens=run.total_tokens,
        latency_ms=run.latency_ms,
        steps_count=run.steps_count,
        reasoning_trace=run.reasoning_trace,
        tools_used=run.tools_used,
        error_message=run.error_message,
        langsmith_run_id=run.langsmith_run_id,
        created_at=run.created_at.isoformat(),
    )