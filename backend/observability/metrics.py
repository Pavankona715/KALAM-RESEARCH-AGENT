"""
Metrics collection — token usage, latency, error rates.

Design rationale:
- MetricsCollector stores metrics in-memory with optional Redis persistence
- Redis is used only if available (reuses the silent-fallback pattern from
  memory/short_term.py) so tests need no Redis
- Token usage is accumulated per-session AND per-user (two buckets)
- Latency histograms are stored as sorted lists for percentile computation
- The /health/readiness endpoint reads these metrics; Prometheus scraping can
  be added by attaching a prometheus_client exporter without changing this module
- All writes are fire-and-forget (no await in hot path) via asyncio.create_task
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class TokenUsageRecord:
    user_id: str
    session_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class LatencyRecord:
    endpoint: str
    duration_ms: float
    status_code: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class ErrorRecord:
    endpoint: str
    error_type: str
    message: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentStepRecord:
    agent_run_id: str
    step_name: str
    duration_ms: float
    success: bool
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# In-memory store (thread-safe via asyncio single-thread assumption)
# ---------------------------------------------------------------------------


class InMemoryMetricsStore:
    def __init__(self) -> None:
        # token_usage[user_id] → list of TokenUsageRecord
        self.token_usage: Dict[str, List[TokenUsageRecord]] = defaultdict(list)
        # session_token_usage[session_id] → accumulated totals
        self.session_totals: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "total": 0}
        )
        # latency[endpoint] → list of duration_ms values (keep last 1000)
        self.latencies: Dict[str, List[float]] = defaultdict(list)
        # errors[endpoint] → count
        self.error_counts: Dict[str, int] = defaultdict(int)
        # agent steps[agent_run_id] → list
        self.agent_steps: Dict[str, List[AgentStepRecord]] = defaultdict(list)

    _MAX_LATENCY_SAMPLES = 1000

    def record_tokens(self, rec: TokenUsageRecord) -> None:
        self.token_usage[rec.user_id].append(rec)
        st = self.session_totals[rec.session_id]
        st["prompt"] += rec.prompt_tokens
        st["completion"] += rec.completion_tokens
        st["total"] += rec.total_tokens

    def record_latency(self, rec: LatencyRecord) -> None:
        bucket = self.latencies[rec.endpoint]
        bucket.append(rec.duration_ms)
        if len(bucket) > self._MAX_LATENCY_SAMPLES:
            self.latencies[rec.endpoint] = bucket[-self._MAX_LATENCY_SAMPLES :]

    def record_error(self, rec: ErrorRecord) -> None:
        self.error_counts[rec.endpoint] += 1

    def record_agent_step(self, rec: AgentStepRecord) -> None:
        self.agent_steps[rec.agent_run_id].append(rec)

    def get_user_token_totals(self, user_id: str) -> Dict[str, int]:
        records = self.token_usage.get(user_id, [])
        totals: Dict[str, int] = {"prompt": 0, "completion": 0, "total": 0}
        for r in records:
            totals["prompt"] += r.prompt_tokens
            totals["completion"] += r.completion_tokens
            totals["total"] += r.total_tokens
        return totals

    def get_session_token_totals(self, session_id: str) -> Dict[str, int]:
        return dict(self.session_totals.get(session_id, {"prompt": 0, "completion": 0, "total": 0}))

    def get_latency_stats(self, endpoint: str) -> Dict[str, float]:
        samples = sorted(self.latencies.get(endpoint, []))
        if not samples:
            return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
        n = len(samples)
        return {
            "count": n,
            "p50": samples[int(n * 0.50)],
            "p95": samples[int(n * 0.95)],
            "p99": samples[int(n * 0.99)],
            "mean": sum(samples) / n,
        }

    def get_error_rate(self, endpoint: Optional[str] = None) -> Dict[str, int]:
        if endpoint:
            return {endpoint: self.error_counts.get(endpoint, 0)}
        return dict(self.error_counts)

    def get_all_stats(self) -> Dict[str, Any]:
        return {
            "endpoints": list(self.latencies.keys()),
            "latency": {ep: self.get_latency_stats(ep) for ep in self.latencies},
            "errors": self.get_error_rate(),
            "sessions_tracked": len(self.session_totals),
            "users_tracked": len(self.token_usage),
        }


# ---------------------------------------------------------------------------
# MetricsCollector (public API)
# ---------------------------------------------------------------------------


class MetricsCollector:
    """
    High-level metrics API.

    All record_* methods are synchronous and never block: Redis writes are
    enqueued as asyncio tasks.

    Usage:
        collector = MetricsCollector()
        collector.record_token_usage("user1", "sess1", "gpt-4o", 100, 200)
        collector.record_request_latency("/chat", 342.1, 200)
    """

    def __init__(self, redis_client=None) -> None:
        self._store = InMemoryMetricsStore()
        self._redis = redis_client  # Optional; injected for persistence

    # ------------------------------------------------------------------
    # Record methods (sync, called in hot path)
    # ------------------------------------------------------------------

    def record_token_usage(
        self,
        user_id: str,
        session_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        rec = TokenUsageRecord(
            user_id=user_id,
            session_id=session_id,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        self._store.record_tokens(rec)
        if self._redis:
            asyncio.create_task(self._persist_tokens(rec))

    def record_request_latency(
        self, endpoint: str, duration_ms: float, status_code: int = 200
    ) -> None:
        rec = LatencyRecord(
            endpoint=endpoint, duration_ms=duration_ms, status_code=status_code
        )
        self._store.record_latency(rec)

    def record_error(self, endpoint: str, error_type: str, message: str = "") -> None:
        rec = ErrorRecord(endpoint=endpoint, error_type=error_type, message=message)
        self._store.record_error(rec)

    def record_agent_step(
        self,
        agent_run_id: str,
        step_name: str,
        duration_ms: float,
        success: bool = True,
    ) -> None:
        rec = AgentStepRecord(
            agent_run_id=agent_run_id,
            step_name=step_name,
            duration_ms=duration_ms,
            success=success,
        )
        self._store.record_agent_step(rec)

    def record_tool_call(self, metrics: "ToolCallMetrics") -> None:
        """Record a tool execution. Called by tools/base.py after every run()."""
        self.record_agent_step(
            agent_run_id=metrics.session_id or "unknown",
            step_name=f"tool:{metrics.tool_name}",
            duration_ms=metrics.latency_ms,
            success=metrics.success,
        )
        if not metrics.success and metrics.error:
            self.record_error(
                endpoint=f"tool:{metrics.tool_name}",
                error_type="tool_error",
                message=metrics.error,
            )

    def record_llm_call(self, metrics: "LLMCallMetrics") -> None:
        """Record an LLM completion. Called by litellm_provider.py after every complete()."""
        if metrics.user_id and metrics.session_id:
            self.record_token_usage(
                user_id=metrics.user_id,
                session_id=metrics.session_id,
                model=metrics.model,
                prompt_tokens=metrics.prompt_tokens,
                completion_tokens=metrics.completion_tokens,
            )
        if not metrics.success and metrics.error:
            self.record_error(
                endpoint=f"llm:{metrics.model}",
                error_type="llm_error",
                message=metrics.error,
            )

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_user_tokens(self, user_id: str) -> Dict[str, int]:
        return self._store.get_user_token_totals(user_id)

    def get_session_tokens(self, session_id: str) -> Dict[str, int]:
        return self._store.get_session_token_totals(session_id)

    def get_latency_stats(self, endpoint: str) -> Dict[str, float]:
        return self._store.get_latency_stats(endpoint)

    def get_all_stats(self) -> Dict[str, Any]:
        return self._store.get_all_stats()

    def get_agent_steps(self, agent_run_id: str) -> List[AgentStepRecord]:
        return self._store.agent_steps.get(agent_run_id, [])

    # ------------------------------------------------------------------
    # Redis persistence helpers (fire-and-forget)
    # ------------------------------------------------------------------

    async def _persist_tokens(self, rec: TokenUsageRecord) -> None:
        try:
            key = f"metrics:tokens:user:{rec.user_id}"
            payload = json.dumps(
                {
                    "session": rec.session_id,
                    "model": rec.model,
                    "prompt": rec.prompt_tokens,
                    "completion": rec.completion_tokens,
                    "total": rec.total_tokens,
                    "ts": rec.timestamp,
                }
            )
            await self._redis.rpush(key, payload)
            await self._redis.expire(key, 60 * 60 * 24 * 30)  # 30-day TTL
        except Exception as exc:  # noqa: BLE001
            logger.debug("MetricsCollector Redis persist failed: %s", exc)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_collector_instance: Optional[MetricsCollector] = None


def get_metrics_collector(redis_client=None) -> MetricsCollector:
    """Return the global MetricsCollector (FastAPI dependency or direct use)."""
    global _collector_instance
    if _collector_instance is None:
        _collector_instance = MetricsCollector(redis_client=redis_client)
    return _collector_instance


# ---------------------------------------------------------------------------
# Backward-compatibility aliases
# (existing modules import LLMCallMetrics and metrics_collector by name)
# ---------------------------------------------------------------------------

@dataclass
class LLMCallMetrics:
    """
    Metrics record for a single LLM call.

    Used by litellm_provider.py to report token usage and latency.
    Token fields are optional so the error path can record without counts.
    """
    model: str
    latency_ms: float
    success: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class ToolCallMetrics:
    """
    Metrics record for a single tool call.

    Used by tools/base.py to report execution time and success.
    """
    tool_name: str
    latency_ms: float
    success: bool
    error: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


# Module-level metrics_collector instance (imported directly by existing modules)
metrics_collector: MetricsCollector = MetricsCollector()