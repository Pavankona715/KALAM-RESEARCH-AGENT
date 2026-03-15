"""
Structured Logging
==================
Configures structured JSON logging for production observability.
In development, outputs human-readable colored logs.
In production, outputs JSON for log aggregation (Datadog, CloudWatch, etc.)

Key design: we use structlog's native PrintLoggerFactory (not stdlib bridge).
This means we must NOT use structlog.stdlib processors (add_log_level,
add_logger_name) — those require a stdlib Logger object with a .name attribute.
Use structlog.processors equivalents instead.
"""

import logging
import sys
from typing import Any

import structlog

from backend.config.settings import get_settings


def configure_logging() -> None:
    """
    Configure structured logging for the entire application.
    Call once at startup in main.py.
    """
    settings = get_settings()

    # Processors compatible with PrintLoggerFactory (no stdlib bridge needed)
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,        # native structlog processor
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.ExceptionRenderer(),
    ]

    if settings.is_development:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]
    else:
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.DEBUG if settings.app_debug else logging.INFO
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,  # False so tests get fresh config each run
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.DEBUG if settings.app_debug else logging.INFO,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a named structured logger.

    Usage:
        logger = get_logger(__name__)
        logger.info("processing request", user_id="123", tokens=456)
    """
    return structlog.get_logger(name)