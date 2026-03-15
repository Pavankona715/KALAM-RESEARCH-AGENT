"""
Agents Endpoint
===============
GET  /agents/runs         — List agent execution history.
GET  /agents/runs/{id}    — Get full details including reasoning trace.
POST /agents/workflow     — Trigger multi-agent workflow directly.
"""

import uuid
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.api.dependencies import CurrentUser, DBSession, Pagination
from backend.db.models.agent_run import AgentRunStatus
from backend.db.repositories.agent_run_repo import agent_run_repo
from backend.llm.router import LLMRouter, get_llm_router
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


class WorkflowRequest(BaseModel):
    request: str = Field(..., min_length=1, max_length=10000,
                         description="The research question or task for the multi-agent workflow")
    session_id: Optional[uuid.UUID] = None

    model_config = {"json_schema_extra": {
        "examples": [{"request": "Write a comprehensive report on the current state of large language models"}]
    }}


class WorkflowResponse(BaseModel):
    run_id: str
    final_report: str
    completed_agents: list[str]
    sources: list[str]
    key_insights: list[str]
    latency_ms: float


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
        db, user.id, skip=pagination.skip, limit=pagination.limit, status=status_enum,
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


@router.post("/workflow", response_model=WorkflowResponse)
async def run_multi_agent_workflow(
    request: WorkflowRequest,
    user: CurrentUser,
    db: DBSession,
    llm: Annotated[LLMRouter, Depends(get_llm_router)],
) -> WorkflowResponse:
    """
    Run the full Planner → Researcher → Analyst → Writer pipeline.

    Use this for complex research tasks that benefit from multi-agent
    coordination: reports, deep research, comparative analysis.

    For simple Q&A, use POST /chat instead.
    """
    import time
    from backend.agents.multi_agent_workflow import MultiAgentWorkflow
    from backend.tools.registry import get_tool_registry

    start = time.perf_counter()
    run_id = str(uuid.uuid4())

    logger.info(
        "workflow_request",
        run_id=run_id,
        user_id=str(user.id),
        request=request.request[:100],
    )

    workflow = MultiAgentWorkflow(llm=llm, tool_registry=get_tool_registry())

    try:
        final_state = await workflow.run(
            user_request=request.request,
            session_id=str(request.session_id or uuid.uuid4()),
            user_id=str(user.id),
            run_id=run_id,
        )
    except Exception as e:
        logger.error("workflow_failed", run_id=run_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Workflow error: {str(e)}",
        )

    latency_ms = (time.perf_counter() - start) * 1000

    return WorkflowResponse(
        run_id=run_id,
        final_report=final_state.get("final_report") or "No report generated.",
        completed_agents=final_state.get("completed_agents", []),
        sources=final_state.get("sources", []),
        key_insights=final_state.get("key_insights", []),
        latency_ms=round(latency_ms, 2),
    )