"""
Test Fixtures
=============
Shared pytest fixtures used across all tests.

Key fixtures:
- app: FastAPI test app with overridden dependencies (no real services needed)
- client: httpx AsyncClient for making test requests
- test_db: In-memory SQLite session for DB tests
- mock_user: A pre-created test user
- mock_llm: A mock LLM router that returns predictable responses
"""

import uuid
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.db.base import Base
from backend.observability.logger import configure_logging

# Configure structlog once at module import so all loggers work in tests.
configure_logging()


# ─── Database ─────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def test_db() -> AsyncGenerator[AsyncSession, None]:
    """
    In-memory SQLite database for tests.
    Creates all tables fresh for each test, drops them after.
    Uses aiosqlite driver — no real PostgreSQL needed.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )

    # Import all models so SQLAlchemy registers them before create_all
    from backend.db.models import (  # noqa: F401
        User, ChatSession, ChatMessage, Document, AgentRun
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


# ─── Mock User ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_user():
    """A fake user object for authentication bypass in tests."""
    from backend.db.models.user import User
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.email = "test@example.com"
    user.is_active = True
    user.is_admin = False
    user.api_key = "test-api-key-12345"
    return user


# ─── Mock LLM ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_response():
    """A predictable LLM response for testing."""
    from backend.llm.base import LLMResponse, TokenUsage
    return LLMResponse(
        content="This is a test response from the mock LLM.",
        usage=TokenUsage(prompt_tokens=50, completion_tokens=20),
        model="gpt-4o-mini",
        finish_reason="stop",
        latency_ms=150.0,
    )


@pytest.fixture
def mock_llm(mock_llm_response):
    """Mock LLM router that returns predictable responses."""
    from backend.llm.router import LLMRouter
    router = MagicMock(spec=LLMRouter)
    router.complete = AsyncMock(return_value=mock_llm_response)
    return router


@pytest.fixture
def mock_agent_factory():
    """Mock AgentFactory whose agent returns a predictable final state."""
    from backend.agents.orchestrator import AgentFactory
    from backend.agents.state import AgentState
    from langchain_core.messages import AIMessage

    mock_state = {
        "messages": [AIMessage(content="This is a test response from the mock agent.")],
        "final_answer": "This is a test response from the mock agent.",
        "step_count": 1,
        "tool_results": [],
        "session_id": "test",
        "user_id": "test",
        "agent_type": "react",
        "max_steps": 10,
        "pending_tool_calls": [],
        "error": None,
        "run_id": None,
        "metadata": {},
    }

    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=mock_state)

    factory = MagicMock(spec=AgentFactory)
    factory.get_agent = MagicMock(return_value=mock_agent)
    return factory


# ─── FastAPI Test App ──────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def app(mock_user, mock_llm, mock_agent_factory, test_db):
    """
    FastAPI test app with all external dependencies mocked.

    Critically: uses lifespan=None to skip startup (avoids hitting
    real PostgreSQL/Redis/Qdrant during tests). All deps are overridden:
      - DB    → in-memory SQLite
      - Auth  → always returns mock_user
      - LLM   → returns predictable mock responses
    """
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from fastapi import Request, status

    from backend.api.middleware import (
        RequestIDMiddleware, RequestLoggingMiddleware,
        TimingMiddleware, setup_cors,
    )
    from backend.api.routes import agents, chat, health, search, upload
    from backend.api.dependencies import get_current_user, get_db_session
    from backend.llm.router import get_llm_router
    from backend.agents.orchestrator import get_agent_factory
    from backend.observability.logger import configure_logging

    configure_logging()

    # Minimal lifespan — no external service connections
    @asynccontextmanager
    async def test_lifespan(app):
        yield

    test_app = FastAPI(
        title="Test App",
        lifespan=test_lifespan,
    )

    setup_cors(test_app, ["*"])
    test_app.add_middleware(RequestLoggingMiddleware)
    test_app.add_middleware(TimingMiddleware)
    test_app.add_middleware(RequestIDMiddleware)

    test_app.include_router(health.router)
    test_app.include_router(chat.router)
    test_app.include_router(upload.router)
    test_app.include_router(search.router)
    test_app.include_router(agents.router)

    @test_app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        errors = [
            {"field": ".".join(str(loc) for loc in e["loc"]), "message": e["msg"]}
            for e in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": "Validation error", "errors": errors},
        )

    # Override all external dependencies
    async def override_db():
        yield test_db

    test_app.dependency_overrides[get_db_session] = override_db
    test_app.dependency_overrides[get_current_user] = lambda: mock_user
    test_app.dependency_overrides[get_llm_router] = lambda: mock_llm
    test_app.dependency_overrides[get_agent_factory] = lambda: mock_agent_factory

    return test_app


@pytest_asyncio.fixture
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP test client with auth header pre-set."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"X-Api-Key": "test-api-key-12345"},
    ) as ac:
        yield ac