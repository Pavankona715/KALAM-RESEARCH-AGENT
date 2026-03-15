"""
Chat Endpoint
=============
POST /chat — Send a message, get an agent response.
GET  /chat/sessions — List user's chat sessions.
GET  /chat/sessions/{session_id} — Get session with message history.
DELETE /chat/sessions/{session_id} — Delete a session.

The actual agent logic lives in backend/agents/ — this route is just
the HTTP adapter. It validates input, calls the agent, persists results.
"""

import time
import uuid
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.api.dependencies import CurrentUser, DBSession, Pagination
from backend.agents.orchestrator import AgentFactory, get_agent_factory
from backend.db.models.chat import MessageRole
from backend.db.repositories.chat_repo import chat_message_repo, chat_session_repo
from backend.llm import get_llm_router
from backend.llm.router import LLMRouter
from backend.observability.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


# ─── Request / Response Schemas ───────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=32_000)
    session_id: Optional[uuid.UUID] = None
    agent_type: str = Field(default="react", pattern="^(react|multi_agent|rag_only)$")
    # Per-request config overrides
    model: Optional[str] = None
    stream: bool = False

    model_config = {"json_schema_extra": {
        "examples": [{
            "message": "What are the latest developments in quantum computing?",
            "agent_type": "react",
        }]
    }}


class MessageResponse(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    model: Optional[str] = None
    tokens_used: int = 0
    latency_ms: float = 0.0


class ChatResponse(BaseModel):
    session_id: uuid.UUID
    message: MessageResponse
    agent_run_id: Optional[uuid.UUID] = None


class SessionSummary(BaseModel):
    id: uuid.UUID
    title: Optional[str]
    agent_type: str
    message_count: int
    total_tokens_used: int
    created_at: str
    updated_at: str


class SessionDetail(BaseModel):
    id: uuid.UUID
    title: Optional[str]
    agent_type: str
    message_count: int
    total_tokens_used: int
    messages: list[MessageResponse]
    created_at: str
    updated_at: str


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(
    request: ChatRequest,
    db: DBSession,
    user: CurrentUser,
    llm: Annotated[LLMRouter, Depends(get_llm_router)],
    agent_factory: Annotated[AgentFactory, Depends(get_agent_factory)],
) -> ChatResponse:
    """
    Send a message to the AI agent and receive a response.

    - Creates a new session if session_id is not provided
    - Persists both user message and assistant response
    - Returns the full assistant response with metadata

    Note: Streaming support (request.stream=True) will be added in Step 4
    when the LLM layer is complete.
    """
    # 1. Get or create session
    if request.session_id:
        session = await chat_session_repo.get_by_id(db, request.session_id)
        if not session or session.user_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {request.session_id} not found",
            )
    else:
        session = await chat_session_repo.create(
            db,
            user_id=user.id,
            agent_type=request.agent_type,
            title=request.message[:80],  # Use first 80 chars as title
        )

    # 2. Persist user message
    user_message = await chat_message_repo.create_message(
        db,
        session_id=session.id,
        role=MessageRole.USER,
        content=request.message,
    )

    # 3. Load conversation history
    history = await chat_message_repo.get_conversation_history(
        db, session.id, limit=20
    )
    history = [h for h in history if h["content"] != request.message]

    # 4. Run ReAct agent
    start_time = time.perf_counter()
    agent = agent_factory.get_agent(request.agent_type)

    try:
        final_state = await agent.run(
            user_message=request.message,
            session_id=str(session.id),
            user_id=str(user.id),
            conversation_history=history,
        )
    except Exception as e:
        logger.error("agent_run_failed", error=str(e), session_id=str(session.id))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Agent error: {str(e)}",
        )

    latency_ms = (time.perf_counter() - start_time) * 1000
    answer = final_state.get("final_answer") or "I could not generate a response."
    steps = final_state.get("step_count", 0)
    tools_used = [r["tool"] for r in final_state.get("tool_results", [])]

    # 5. Persist assistant response
    assistant_message = await chat_message_repo.create_message(
        db,
        session_id=session.id,
        role=MessageRole.ASSISTANT,
        content=answer,
        model="agent",
        prompt_tokens=0,
        completion_tokens=0,
        latency_ms=latency_ms,
    )

    # 6. Update session counters
    await chat_session_repo.increment_counters(
        db, session.id, tokens=0, messages=2
    )

    logger.info(
        "chat_request",
        session_id=str(session.id),
        user_id=str(user.id),
        agent_type=request.agent_type,
        message_length=len(request.message),
        steps=steps,
        tools_used=tools_used,
        latency_ms=round(latency_ms, 2),
    )

    return ChatResponse(
        session_id=session.id,
        message=MessageResponse(
            id=assistant_message.id,
            role=assistant_message.role,
            content=assistant_message.content,
            model=assistant_message.model,
            tokens_used=0,
            latency_ms=round(latency_ms, 2),
        ),
    )


@router.get("/sessions", response_model=list[SessionSummary])
async def list_sessions(
    db: DBSession,
    user: CurrentUser,
    pagination: Pagination,
) -> list[SessionSummary]:
    """List all chat sessions for the authenticated user."""
    sessions = await chat_session_repo.get_sessions_for_user(
        db, user.id, skip=pagination.skip, limit=pagination.limit
    )
    return [
        SessionSummary(
            id=s.id,
            title=s.title,
            agent_type=s.agent_type,
            message_count=s.message_count,
            total_tokens_used=s.total_tokens_used,
            created_at=s.created_at.isoformat(),
            updated_at=s.updated_at.isoformat(),
        )
        for s in sessions
    ]


@router.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_session(
    session_id: uuid.UUID,
    db: DBSession,
    user: CurrentUser,
) -> SessionDetail:
    """Get a specific session with its full message history."""
    session = await chat_session_repo.get_with_messages(db, session_id)

    if not session or session.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    return SessionDetail(
        id=session.id,
        title=session.title,
        agent_type=session.agent_type,
        message_count=session.message_count,
        total_tokens_used=session.total_tokens_used,
        messages=[
            MessageResponse(
                id=m.id,
                role=m.role,
                content=m.content,
                model=m.model,
                tokens_used=m.prompt_tokens + m.completion_tokens,
                latency_ms=m.latency_ms,
            )
            for m in session.messages
        ],
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: uuid.UUID,
    db: DBSession,
    user: CurrentUser,
) -> None:
    """Delete a chat session and all its messages (cascade)."""
    session = await chat_session_repo.get_by_id(db, session_id)

    if not session or session.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    await chat_session_repo.delete(db, session_id)
    logger.info("session_deleted", session_id=str(session_id), user_id=str(user.id))