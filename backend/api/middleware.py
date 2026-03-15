"""
API Middleware
==============
Middleware runs on every request before route handlers.

Stack (applied in reverse order of registration):
1. CORS — handle preflight requests
2. RequestID — inject X-Request-ID for distributed tracing
3. Timing — measure and attach X-Process-Time
4. RequestLogging — structured access log per request

Each middleware is a separate class — easy to enable/disable individually.
"""

import time
import uuid

from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

import structlog

from backend.observability.logger import get_logger

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Injects a unique request ID into every request.
    - Reads X-Request-ID header if provided (for client-side correlation)
    - Generates a new UUID if not provided
    - Adds it to structlog context so all logs within the request include it
    - Returns it in the response header for client correlation
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Bind to structlog context — all subsequent log calls in this request
        # will automatically include request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """
    Measures request processing time and attaches it to the response.
    X-Process-Time header contains duration in milliseconds.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time"] = f"{elapsed_ms:.2f}ms"
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Structured access log for every HTTP request.
    Logs method, path, status code, and duration.
    Skips /health to avoid log spam.
    """

    SKIP_PATHS = {"/health", "/favicon.ico"}

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
            client_ip=request.client.host if request.client else "unknown",
        )

        return response


def setup_cors(app, allowed_origins: list[str]) -> None:
    """Configure CORS middleware."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time"],
    )