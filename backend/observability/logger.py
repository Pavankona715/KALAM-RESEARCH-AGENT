"""
Structured logging configuration.

Design rationale:
- All log records are emitted as JSON so they can be ingested by any log
  aggregator (Datadog, Loki, CloudWatch) without parsing rules
- RequestID and TraceID are injected automatically via a logging.Filter that
  reads from contextvars — same IDs that tracer.py sets
- configure_logging() is idempotent and safe to call from conftest.py before
  any other imports
- Log level is controlled by LOG_LEVEL env var (default INFO)
"""

from __future__ import annotations

import json
import logging
import logging.config
import os
import time
from contextvars import ContextVar
from typing import Optional

# ---------------------------------------------------------------------------
# Context variables (set by RequestLogging middleware)
# ---------------------------------------------------------------------------

_request_id: ContextVar[Optional[str]] = ContextVar("_request_id", default=None)
_trace_id: ContextVar[Optional[str]] = ContextVar("_log_trace_id", default=None)
_user_id: ContextVar[Optional[str]] = ContextVar("_log_user_id", default=None)


def set_log_context(
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> None:
    """Set log context vars for the current async task."""
    if request_id is not None:
        _request_id.set(request_id)
    if trace_id is not None:
        _trace_id.set(trace_id)
    if user_id is not None:
        _user_id.set(user_id)


def clear_log_context() -> None:
    _request_id.set(None)
    _trace_id.set(None)
    _user_id.set(None)


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects.

    Fields always present:
        timestamp, level, logger, message
    Fields when available (from context vars or LogRecord extras):
        request_id, trace_id, user_id, exc_info, duration_ms
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Inject context vars
        rid = _request_id.get()
        tid = _trace_id.get()
        uid = _user_id.get()
        if rid:
            log_obj["request_id"] = rid
        if tid:
            log_obj["trace_id"] = tid
        if uid:
            log_obj["user_id"] = uid

        # Extra fields attached via logger.info("...", extra={...})
        for key in ("duration_ms", "endpoint", "status_code", "agent_run_id", "step"):
            val = getattr(record, key, None)
            if val is not None:
                log_obj[key] = val

        # Exception info
        if record.exc_info:
            log_obj["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, ensure_ascii=False)


# ---------------------------------------------------------------------------
# configure_logging (idempotent)
# ---------------------------------------------------------------------------

_configured = False


def configure_logging(log_level: Optional[str] = None) -> None:
    """
    Configure root logger with JSON output.

    Safe to call multiple times — subsequent calls are no-ops.
    Call from conftest.py at module level before any other imports.
    """
    global _configured
    if _configured:
        return
    _configured = True

    level = (log_level or os.getenv("LOG_LEVEL", "INFO")).upper()

    formatter = JSONFormatter()
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Quieten noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging configured", extra={"log_level": level}
    )


# ---------------------------------------------------------------------------
# Agent reasoning trace logger
# ---------------------------------------------------------------------------


class ReasoningTraceLogger:
    """
    Records agent reasoning steps to the AgentRun table.

    Kept separate from the main log stream so reasoning traces can be
    queried independently via the /agents/{run_id}/trace endpoint.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger("agent.reasoning")

    def log_step(
        self,
        agent_run_id: str,
        step_type: str,
        content: str,
        tool_name: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        self._logger.info(
            "agent_step",
            extra={
                "agent_run_id": agent_run_id,
                "step_type": step_type,
                "content": content[:2000],  # truncate to avoid huge logs
                "tool_name": tool_name,
                "duration_ms": duration_ms,
            },
        )

    def log_final_answer(
        self,
        agent_run_id: str,
        answer: str,
        total_steps: int,
        total_duration_ms: float,
    ) -> None:
        self._logger.info(
            "agent_final_answer",
            extra={
                "agent_run_id": agent_run_id,
                "answer_length": len(answer),
                "total_steps": total_steps,
                "total_duration_ms": total_duration_ms,
            },
        )


_reasoning_logger: Optional[ReasoningTraceLogger] = None


def get_reasoning_logger() -> ReasoningTraceLogger:
    global _reasoning_logger
    if _reasoning_logger is None:
        _reasoning_logger = ReasoningTraceLogger()
    return _reasoning_logger


def get_logger(name: str) -> "StructuredLogger":
    """
    Return a StructuredLogger by name.

    Drop-in replacement that supports structured keyword arguments:
        logger.info("event_name", key=value, other=value)

    This matches the calling convention used throughout the codebase
    (short_term.py, manager.py, qdrant_adapter.py, rag/pipeline.py, etc.)
    """
    return StructuredLogger(name)


class StructuredLogger:
    """
    Thin wrapper around stdlib logging.Logger that accepts keyword
    arguments and serialises them into the log message.

    Usage (existing codebase pattern):
        logger = get_logger(__name__)
        logger.info("qdrant_client_initialized", mode="in_memory")
        logger.warning("tool_not_found", tool_name=name)
        logger.error("query_embedding_failed", error=str(e))
    """

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    def _format(self, msg: str, kwargs: dict) -> str:
        if not kwargs:
            return msg
        pairs = " ".join(f"{k}={v!r}" for k, v in kwargs.items())
        return f"{msg} {pairs}"

    def debug(self, msg: str, *args, **kwargs) -> None:
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug(self._format(msg, kwargs), *args)

    def info(self, msg: str, *args, **kwargs) -> None:
        if self._logger.isEnabledFor(logging.INFO):
            self._logger.info(self._format(msg, kwargs), *args)

    def warning(self, msg: str, *args, **kwargs) -> None:
        if self._logger.isEnabledFor(logging.WARNING):
            self._logger.warning(self._format(msg, kwargs), *args)

    def error(self, msg: str, *args, **kwargs) -> None:
        if self._logger.isEnabledFor(logging.ERROR):
            self._logger.error(self._format(msg, kwargs), *args)

    def critical(self, msg: str, *args, **kwargs) -> None:
        if self._logger.isEnabledFor(logging.CRITICAL):
            self._logger.critical(self._format(msg, kwargs), *args)

    def exception(self, msg: str, *args, **kwargs) -> None:
        self._logger.exception(self._format(msg, kwargs), *args)