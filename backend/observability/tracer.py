"""
Distributed tracing — LangSmith + OpenTelemetry.

Design rationale:
- Two independent tracing back-ends coexist: LangSmith (LLM-native, free tier)
  and OpenTelemetry (infra-standard, exportable to Jaeger/OTLP/Datadog).
- Both are entirely optional: if their SDK is absent or not configured they
  silently become no-ops.  Business logic never imports these SDKs directly.
- AgentTracer is the single public API used by routes and agents.
- All context propagation uses Python contextvars so it's async-safe and
  works across LangGraph nodes without explicit passing.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Context variables (async-safe trace context)
# ---------------------------------------------------------------------------

_current_trace_id: ContextVar[Optional[str]] = ContextVar(
    "_current_trace_id", default=None
)
_current_span_id: ContextVar[Optional[str]] = ContextVar(
    "_current_span_id", default=None
)


def get_current_trace_id() -> Optional[str]:
    return _current_trace_id.get()


def get_current_span_id() -> Optional[str]:
    return _current_span_id.get()


# ---------------------------------------------------------------------------
# Span data model
# ---------------------------------------------------------------------------


@dataclass
class SpanData:
    name: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    start_time: float
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000


# ---------------------------------------------------------------------------
# LangSmith integration (optional)
# ---------------------------------------------------------------------------


class _LangSmithTracer:
    """
    Thin wrapper around langsmith.Client.

    Falls back to a no-op if langsmith is not installed or LANGCHAIN_API_KEY
    is not set.
    """

    def __init__(self) -> None:
        self._client = None
        self._enabled = False
        self._try_init()

    def _try_init(self) -> None:
        api_key = os.getenv("LANGCHAIN_API_KEY", "")
        if not api_key:
            return
        try:
            import langsmith  # noqa: F401
            from langsmith import Client

            self._client = Client(api_key=api_key)
            self._enabled = True
            logger.info("LangSmith tracing enabled")
        except ImportError:
            logger.debug("langsmith not installed — LangSmith tracing disabled")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def create_run(
        self,
        name: str,
        run_type: str,
        inputs: Dict[str, Any],
        trace_id: Optional[str] = None,
        parent_run_id: Optional[str] = None,
    ) -> Optional[str]:
        """Create a LangSmith run. Returns run_id or None."""
        if not self._enabled or not self._client:
            return None
        try:
            run_id = str(uuid.uuid4())
            self._client.create_run(
                name=name,
                run_type=run_type,
                inputs=inputs,
                id=run_id,
                trace_id=trace_id,
                parent_run_id=parent_run_id,
            )
            return run_id
        except Exception as exc:  # noqa: BLE001
            logger.debug("LangSmith create_run failed: %s", exc)
            return None

    def end_run(
        self,
        run_id: str,
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        if not self._enabled or not self._client or not run_id:
            return
        try:
            self._client.update_run(
                run_id,
                outputs=outputs or {},
                error=error,
                end_time=time.time(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("LangSmith end_run failed: %s", exc)


# ---------------------------------------------------------------------------
# OpenTelemetry integration (optional)
# ---------------------------------------------------------------------------


class _OTelTracer:
    """
    Thin wrapper around opentelemetry-api.

    Falls back to no-op if opentelemetry is not installed or
    OTEL_EXPORTER_OTLP_ENDPOINT is not set.
    """

    def __init__(self) -> None:
        self._tracer = None
        self._enabled = False
        self._try_init()

    def _try_init(self) -> None:
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        service_name = os.getenv("OTEL_SERVICE_NAME", "universal-ai-agent")
        if not endpoint:
            return
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            resource = Resource(attributes={"service.name": service_name})
            provider = TracerProvider(resource=resource)
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
            )
            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer(service_name)
            self._enabled = True
            logger.info("OpenTelemetry tracing enabled → %s", endpoint)
        except ImportError:
            logger.debug("opentelemetry not installed — OTel tracing disabled")
        except Exception as exc:  # noqa: BLE001
            logger.warning("OTel init failed: %s", exc)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @contextlib.contextmanager
    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        if not self._enabled or not self._tracer:
            yield None
            return
        with self._tracer.start_as_current_span(name) as span:
            if attributes:
                for k, v in attributes.items():
                    span.set_attribute(k, str(v))
            yield span


# ---------------------------------------------------------------------------
# Public AgentTracer
# ---------------------------------------------------------------------------


class AgentTracer:
    """
    Single tracing facade used by routes and agents.

    Usage:
        tracer = AgentTracer()

        async with tracer.trace_request("chat", user_id="u1", session_id="s1") as ctx:
            ...
            async with tracer.trace_agent_step("reasoning", ctx["trace_id"]):
                ...
    """

    def __init__(self) -> None:
        self._langsmith = _LangSmithTracer()
        self._otel = _OTelTracer()

    # ------------------------------------------------------------------
    # High-level context managers
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def trace_request(
        self,
        endpoint: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Trace an entire HTTP request.

        Yields a context dict: {"trace_id": str, "run_id": Optional[str]}
        """
        trace_id = str(uuid.uuid4())
        token = _current_trace_id.set(trace_id)
        start = time.time()
        attributes = {
            "endpoint": endpoint,
            "user_id": user_id or "anonymous",
            "session_id": session_id or "",
            **(metadata or {}),
        }

        run_id = self._langsmith.create_run(
            name=f"request:{endpoint}",
            run_type="chain",
            inputs=attributes,
            trace_id=trace_id,
        )

        ctx = {"trace_id": trace_id, "run_id": run_id}
        error: Optional[str] = None

        try:
            with self._otel.start_span(f"request:{endpoint}", attributes):
                yield ctx
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            duration_ms = (time.time() - start) * 1000
            logger.debug(
                "trace_request endpoint=%s trace_id=%s duration_ms=%.1f error=%s",
                endpoint,
                trace_id,
                duration_ms,
                error,
            )
            if run_id:
                self._langsmith.end_run(
                    run_id,
                    outputs={"duration_ms": duration_ms},
                    error=error,
                )
            _current_trace_id.reset(token)

    @asynccontextmanager
    async def trace_agent_step(
        self,
        step_name: str,
        trace_id: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Trace a single agent reasoning/tool step.

        Yields {"span_id": str, "run_id": Optional[str]}
        """
        span_id = str(uuid.uuid4())
        parent_span = _current_span_id.get()
        token = _current_span_id.set(span_id)
        resolved_trace = trace_id or _current_trace_id.get() or str(uuid.uuid4())
        start = time.time()

        run_id = self._langsmith.create_run(
            name=f"step:{step_name}",
            run_type="tool",
            inputs=inputs or {},
            trace_id=resolved_trace,
            parent_run_id=parent_span,
        )

        error: Optional[str] = None
        ctx = {"span_id": span_id, "run_id": run_id}

        try:
            with self._otel.start_span(f"step:{step_name}", metadata):
                yield ctx
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            duration_ms = (time.time() - start) * 1000
            logger.debug(
                "trace_agent_step step=%s span_id=%s duration_ms=%.1f",
                step_name,
                span_id,
                duration_ms,
            )
            if run_id:
                self._langsmith.end_run(
                    run_id,
                    outputs={"duration_ms": duration_ms},
                    error=error,
                )
            _current_span_id.reset(token)

    @asynccontextmanager
    async def trace_llm_call(
        self,
        model: str,
        prompt_tokens: int = 0,
        trace_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Trace a single LLM call. Yields context dict."""
        async with self.trace_agent_step(
            f"llm:{model}",
            trace_id=trace_id,
            inputs={"model": model, "prompt_tokens": prompt_tokens},
        ) as ctx:
            yield ctx

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def langsmith_enabled(self) -> bool:
        return self._langsmith.enabled

    @property
    def otel_enabled(self) -> bool:
        return self._otel.enabled


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_tracer_instance: Optional[AgentTracer] = None


def get_tracer() -> AgentTracer:
    """Return the singleton AgentTracer (FastAPI dependency or direct use)."""
    global _tracer_instance
    if _tracer_instance is None:
        _tracer_instance = AgentTracer()
    return _tracer_instance


def configure_tracing() -> AgentTracer:
    """
    Backward-compatibility alias for main.py lifespan startup.

    Initialises the global AgentTracer (LangSmith + OTel) and returns it.
    Both back-ends are optional — missing env vars produce silent no-ops.
    """
    return get_tracer()