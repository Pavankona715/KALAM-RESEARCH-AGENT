"""
Metrics Tracking
================
Tracks key system metrics: token usage, latency, errors, tool calls.
Designed to be backend-agnostic - swap Prometheus for StatsD by changing the sink.
"""

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

from backend.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LLMCallMetrics:
    """Metrics for a single LLM API call."""
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    agent_name: Optional[str] = None


@dataclass
class ToolCallMetrics:
    """Metrics for a single tool invocation."""
    tool_name: str
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


@dataclass
class SessionMetrics:
    """Aggregated metrics for a single user session/request."""
    session_id: str
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    llm_calls: list[LLMCallMetrics] = field(default_factory=list)
    tool_calls: list[ToolCallMetrics] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_llm_call(self, metrics: LLMCallMetrics) -> None:
        self.llm_calls.append(metrics)
        self.total_tokens += metrics.total_tokens
        self.total_latency_ms += metrics.latency_ms
        if not metrics.success and metrics.error:
            self.errors.append(f"LLM error: {metrics.error}")

    def add_tool_call(self, metrics: ToolCallMetrics) -> None:
        self.tool_calls.append(metrics)
        self.total_latency_ms += metrics.latency_ms
        if not metrics.success and metrics.error:
            self.errors.append(f"Tool error ({metrics.tool_name}): {metrics.error}")

    def to_summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "total_tokens": self.total_tokens,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "llm_calls_count": len(self.llm_calls),
            "tool_calls_count": len(self.tool_calls),
            "error_count": len(self.errors),
            "models_used": list({m.model for m in self.llm_calls}),
            "tools_used": list({t.tool_name for t in self.tool_calls}),
        }


class MetricsCollector:
    """
    Collects and emits system metrics.
    
    In production, extend this to push to Prometheus, Datadog, etc.
    """

    def record_llm_call(self, metrics: LLMCallMetrics) -> None:
        """Record metrics from an LLM API call."""
        logger.info(
            "llm_call",
            model=metrics.model,
            prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.completion_tokens,
            total_tokens=metrics.total_tokens,
            latency_ms=round(metrics.latency_ms, 2),
            success=metrics.success,
            agent=metrics.agent_name,
        )

    def record_tool_call(self, metrics: ToolCallMetrics) -> None:
        """Record metrics from a tool invocation."""
        logger.info(
            "tool_call",
            tool=metrics.tool_name,
            latency_ms=round(metrics.latency_ms, 2),
            success=metrics.success,
        )

    def record_error(self, error_type: str, message: str, **context) -> None:
        """Record an application error."""
        logger.error("app_error", error_type=error_type, message=message, **context)

    @asynccontextmanager
    async def measure_latency(
        self, operation: str, **tags
    ) -> AsyncGenerator[dict, None]:
        """
        Context manager to measure operation latency.
        
        Usage:
            async with metrics.measure_latency("vector_search", collection="docs") as m:
                results = await vector_db.search(query)
            # m["latency_ms"] is now populated
        """
        start = time.perf_counter()
        result = {"latency_ms": 0.0}
        try:
            yield result
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            result["latency_ms"] = elapsed_ms
            logger.debug(
                "operation_latency",
                operation=operation,
                latency_ms=round(elapsed_ms, 2),
                **tags,
            )


# Module-level singleton
metrics_collector = MetricsCollector()