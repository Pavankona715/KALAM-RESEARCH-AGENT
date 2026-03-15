"""
FastAPI Application
===================
Main application entrypoint. Defines:
- App creation with lifespan context manager
- Middleware stack
- Route registration
- Global exception handlers

The lifespan context manager handles startup/shutdown:
- Startup: Initialize DB pool, Redis, observability
- Shutdown: Gracefully close all connections

Run with: uvicorn backend.api.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.api.middleware import (
    RequestIDMiddleware,
    RequestLoggingMiddleware,
    TimingMiddleware,
    setup_cors,
)
from backend.api.routes import agents, chat, health, search, upload
from backend.config.settings import get_settings
from backend.observability.logger import configure_logging, get_logger
from backend.observability.tracer import configure_tracing

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Startup sequence:
    1. Configure logging (must be first — everything else logs)
    2. Configure tracing (LangSmith, OTEL)
    3. Initialize PostgreSQL connection pool
    4. Verify Redis connectivity
    5. Verify Qdrant connectivity

    Shutdown sequence (reverse order):
    1. Close DB connections
    2. Flush remaining metrics/traces
    """
    settings = get_settings()

    # ── Startup ──────────────────────────────────────────────────────────────
    configure_logging()
    logger.info("application_starting", env=settings.app_env.value)

    configure_tracing()

    # Initialize PostgreSQL
    try:
        from backend.db.session import init_db
        await init_db()
        logger.info("postgres_ready")
    except Exception as e:
        logger.error("postgres_init_failed", error=str(e))
        raise  # Fatal — can't start without DB

    # Verify Redis (non-fatal in development)
    try:
        import redis.asyncio as aioredis
        client = aioredis.from_url(settings.redis_url, socket_timeout=3)
        await client.ping()
        await client.aclose()
        logger.info("redis_ready")
    except Exception as e:
        if settings.is_production:
            raise
        logger.warning("redis_unavailable", error=str(e), note="continuing in dev mode")

    # Verify Qdrant (non-fatal in development)
    try:
        from qdrant_client import AsyncQdrantClient
        qclient = AsyncQdrantClient(url=settings.qdrant_url, timeout=3)
        await qclient.get_collections()
        await qclient.close()
        logger.info("qdrant_ready")
    except Exception as e:
        if settings.is_production:
            raise
        logger.warning("qdrant_unavailable", error=str(e), note="continuing in dev mode")

    logger.info(
        "application_ready",
        port=settings.app_port,
        providers=settings.configured_llm_providers,
    )

    yield  # ← Application runs here

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("application_shutting_down")

    from backend.db.session import close_db
    await close_db()

    logger.info("application_stopped")


def create_app() -> FastAPI:
    """
    Application factory — creates and configures the FastAPI app.
    Using a factory function (instead of module-level app) makes testing easier.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description=(
            "Production-grade AI Research Agent with RAG, multi-agent workflows, "
            "web search, document ingestion, and MCP integrations."
        ),
        version="0.1.0",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── Middleware (outermost first) ──────────────────────────────────────────
    # Note: FastAPI applies middleware in REVERSE registration order
    # So last registered = outermost (first to run)

    # CORS must be outermost (handles OPTIONS preflight before any auth)
    cors_origins = (
        ["*"] if settings.is_development
        else ["https://yourdomain.com"]  # Tighten in production
    )
    setup_cors(app, cors_origins)

    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIDMiddleware)  # innermost — runs first

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(upload.router)
    app.include_router(search.router)
    app.include_router(agents.router)

    # ── Exception Handlers ────────────────────────────────────────────────────
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Return clean validation errors instead of FastAPI's verbose default."""
        errors = []
        for error in exc.errors():
            errors.append({
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
            })
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": "Validation error", "errors": errors},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all handler — never expose internal errors to clients."""
        logger.error(
            "unhandled_exception",
            path=request.url.path,
            method=request.method,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )

    return app


# Module-level app instance (used by uvicorn)
app = create_app()