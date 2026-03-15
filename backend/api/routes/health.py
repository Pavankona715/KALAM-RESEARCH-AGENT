"""
Health Check Endpoint
=====================
GET /health — liveness probe (is the process alive?)
GET /health/ready — readiness probe (can the service handle traffic?)

Liveness: Used by process managers to decide whether to restart the process.
Readiness: Used by load balancers to decide whether to route traffic here.

These are different. A service can be alive but not yet ready
(still connecting to DB at startup).
"""

import time
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.config.settings import get_settings
from backend.observability.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Health"])

# Application start time for uptime calculation
_start_time = time.time()


class ServiceStatus(BaseModel):
    name: str
    status: Literal["ok", "degraded", "down"]
    latency_ms: float | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    version: str = "0.1.0"
    environment: str
    uptime_seconds: float
    services: list[ServiceStatus] = []


async def check_postgres() -> ServiceStatus:
    """Verify PostgreSQL connectivity."""
    start = time.perf_counter()
    try:
        from backend.db.session import get_engine
        import sqlalchemy
        async with get_engine().connect() as conn:
            await conn.execute(sqlalchemy.text("SELECT 1"))
        latency = (time.perf_counter() - start) * 1000
        return ServiceStatus(name="postgresql", status="ok", latency_ms=round(latency, 2))
    except Exception as e:
        return ServiceStatus(name="postgresql", status="down", error=str(e)[:100])


async def check_redis() -> ServiceStatus:
    """Verify Redis connectivity."""
    start = time.perf_counter()
    try:
        import redis.asyncio as aioredis
        settings = get_settings()
        client = aioredis.from_url(settings.redis_url, socket_timeout=2)
        await client.ping()
        await client.aclose()
        latency = (time.perf_counter() - start) * 1000
        return ServiceStatus(name="redis", status="ok", latency_ms=round(latency, 2))
    except Exception as e:
        return ServiceStatus(name="redis", status="down", error=str(e)[:100])


async def check_qdrant() -> ServiceStatus:
    """Verify Qdrant connectivity."""
    start = time.perf_counter()
    try:
        from qdrant_client import AsyncQdrantClient
        settings = get_settings()
        client = AsyncQdrantClient(url=settings.qdrant_url, timeout=2)
        await client.get_collections()
        await client.close()
        latency = (time.perf_counter() - start) * 1000
        return ServiceStatus(name="qdrant", status="ok", latency_ms=round(latency, 2))
    except Exception as e:
        return ServiceStatus(name="qdrant", status="down", error=str(e)[:100])


@router.get("/health", response_model=HealthResponse)
async def liveness():
    """
    Liveness probe — confirms the process is running.
    Does NOT check external dependencies (fast, always responds).
    """
    settings = get_settings()
    return HealthResponse(
        status="ok",
        environment=settings.app_env.value,
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@router.get("/health/ready", response_model=HealthResponse)
async def readiness():
    """
    Readiness probe — confirms all dependencies are reachable.
    Returns 503 if any critical service is down.
    """
    import asyncio
    settings = get_settings()

    # Check all services concurrently
    postgres, redis, qdrant = await asyncio.gather(
        check_postgres(),
        check_redis(),
        check_qdrant(),
    )

    services = [postgres, redis, qdrant]
    has_down = any(s.status == "down" for s in services)
    overall_status = "down" if has_down else "ok"

    return HealthResponse(
        status=overall_status,
        environment=settings.app_env.value,
        uptime_seconds=round(time.time() - _start_time, 2),
        services=services,
    )