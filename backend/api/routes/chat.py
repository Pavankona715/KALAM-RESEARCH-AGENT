"""
Chat Endpoint
=============
POST /chat — Send a message, get an agent response.
GET  /chat/sessions — List user's chat sessions.
GET  /chat/sessions/{session_id} — Get session with message history.
DELETE /chat/sessions/{session_id} — Delete a session.

The actual agent logic lives in backend/agents/ — this route is just
the HTTP adapter. It validates input, calls the agent, persists results.

Step 14 additions:
- InputValidator  — prompt injection / content policy check before agent runs
- OutputValidator — leakage / PII check on the assistant reply
- PermissionChecker — filters the tool allowlist to what the user's role permits
- AgentTracer     — LangSmith + OTel spans around the full request and sub-steps
- MetricsCollector — token usage + latency + error rate tracking
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
from backend.guardrails.input_validator import InputValidator, get_input_validator
from backend.guardrails.output_validator import OutputValidator, get_output_validator
from backend.guardrails.permissions import PermissionChecker, get_permission_checker
from backend.llm import get_llm_router
from backend.llm.router import LLMRouter
from backend.memory.manager import MemoryManager, get_memory_manager
from backend.observability.logger import get_logger
from backend.observability.metrics import MetricsCollector, get_metrics_collector
from backend.observability.tracer import AgentTracer, get_tracer
from backend.rag.retriever import get_retriever

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
    # Step 14 additions — observability / guardrail metadata
    trace_id: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)


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
    memory: Annotated["MemoryManager", Depends(get_memory_manager)],
    # ── Step 14: guardrails + observability (all have module-level fallbacks) ──
    input_validator: Annotated[InputValidator, Depends(get_input_validator)],
    output_validator: Annotated[OutputValidator, Depends(get_output_validator)],
    permission_checker: Annotated[PermissionChecker, Depends(get_permission_checker)],
    tracer: Annotated[AgentTracer, Depends(get_tracer)],
    metrics: Annotated[MetricsCollector, Depends(get_metrics_collector)],
) -> ChatResponse:
    """
    Send a message to the AI agent and receive a response.

    - Creates a new session if session_id is not provided
    - Persists both user message and assistant response
    - Returns the full assistant response with metadata
    - Validates input for injection attacks before agent runs
    - Validates output for leakage/PII before returning
    - Enforces tool permissions based on user role
    - Traces the full request in LangSmith/OTel
    - Records token usage and latency metrics
    """
    warnings: list[str] = []

    # ── 1. Input validation (guardrail) ──────────────────────────────────────
    validation = input_validator.validate(request.message)
    if validation.should_block:
        logger.warning(
            "input_blocked",
            user_id=str(user.id),
            risk_score=validation.risk_score,
            violations=validation.violations,
        )
        metrics.record_error("/chat", "input_blocked")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "input_rejected",
                "reason": "Request contains disallowed content",
                "violations": validation.violations,
            },
        )
    if validation.risk_score > 0.3:
        warnings.append(f"input_risk_score:{validation.risk_score:.2f}")

    # ── 2. Permission check — filter tools to user's role allowlist ───────────
    user_role = getattr(user, "role", "free")
    allowed_tools = permission_checker.get_allowed_tools(user_role)

    # ── 3. Wrap the entire request in a trace span ────────────────────────────
    async with tracer.trace_request(
        endpoint="/chat",
        user_id=str(user.id),
        session_id=str(request.session_id) if request.session_id else None,
    ) as trace_ctx:
        trace_id: str = trace_ctx["trace_id"]

        # ── 4. Get or create session ──────────────────────────────────────────
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
                title=request.message[:80],
            )

        # ── 5. Persist user message ───────────────────────────────────────────
        user_message = await chat_message_repo.create_message(
            db,
            session_id=session.id,
            role=MessageRole.USER,
            content=request.message,
        )

        # ── 6. Load memory context ────────────────────────────────────────────
        async with tracer.trace_agent_step("load_memory", trace_id=trace_id):
            memory_context = await memory.load_context(
                session_id=str(session.id),
                user_id=str(user.id),
                current_query=request.message,
            )

        # ── 7. RAG retrieval ──────────────────────────────────────────────────
        async with tracer.trace_agent_step("rag_retrieval", trace_id=trace_id):
            retriever = get_retriever()
            try:
                retrieved_docs = await retriever.retrieve_for_agent(
                    query=request.message,
                    session_id=str(session.id),
                    top_k=5,
                )
            except Exception as exc:
                logger.warning("rag_retrieval_failed", error=str(exc), session_id=str(session.id))
                retrieved_docs = []

        # ── 8. Run agent ──────────────────────────────────────────────────────
        start_time = time.perf_counter()
        agent = agent_factory.get_agent(request.agent_type)

        async with tracer.trace_agent_step("agent_execution", trace_id=trace_id):
            try:
                final_state = await agent.run(
                    user_message=request.message,
                    session_id=str(session.id),
                    user_id=str(user.id),
                    conversation_history=memory_context.conversation_history,
                    retrieved_docs=retrieved_docs,
                    # Step 14: pass role-filtered tool list to the agent
                    allowed_tools=allowed_tools,
                )
            except Exception as exc:
                logger.error(
                    "agent_run_failed",
                    error=str(exc),
                    session_id=str(session.id),
                )
                metrics.record_error("/chat", type(exc).__name__)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Agent error: {str(exc)}",
                )

        latency_ms = (time.perf_counter() - start_time) * 1000
        answer = final_state.get("final_answer") or "I could not generate a response."
        steps = final_state.get("step_count", 0)
        tools_used = [r["tool"] for r in final_state.get("tool_results", [])]
        usage = final_state.get("usage") or {}

        # ── 9. Output validation (guardrail) ──────────────────────────────────
        out_result = output_validator.validate_text(answer)
        if not out_result.valid:
            logger.warning(
                "output_validation_failed",
                errors=out_result.errors,
                session_id=str(session.id),
            )
            answer = "I'm sorry, I was unable to generate a valid response. Please try again."
            warnings.append("output_validation_failed")
        else:
            answer = out_result.raw_output
            if out_result.was_sanitized:
                warnings.append("output_sanitized")
            warnings.extend(out_result.warnings)

        # ── 10. Save turn to memory (Redis short-term) ────────────────────────
        await memory.save_turn(
            session_id=str(session.id),
            user_id=str(user.id),
            user_message=request.message,
            assistant_response=answer,
        )

        # ── 11. Persist assistant response to PostgreSQL ──────────────────────
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        assistant_message = await chat_message_repo.create_message(
            db,
            session_id=session.id,
            role=MessageRole.ASSISTANT,
            content=answer,
            model=request.model or "agent",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )

        # ── 12. Update session counters ───────────────────────────────────────
        total_tokens = prompt_tokens + completion_tokens
        await chat_session_repo.increment_counters(
            db, session.id, tokens=total_tokens, messages=2
        )

    # ── 13. Record metrics (outside trace span — fire and forget) ─────────────
    metrics.record_request_latency("/chat", latency_ms)
    if usage:
        metrics.record_token_usage(
            user_id=str(user.id),
            session_id=str(session.id),
            model=request.model or "agent",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
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
        trace_id=trace_id,
        warnings=warnings,
    )

    return ChatResponse(
        session_id=session.id,
        message=MessageResponse(
            id=assistant_message.id,
            role=assistant_message.role,
            content=assistant_message.content,
            model=assistant_message.model,
            tokens_used=total_tokens,
            latency_ms=round(latency_ms, 2),
        ),
        trace_id=trace_id,
        warnings=warnings,
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