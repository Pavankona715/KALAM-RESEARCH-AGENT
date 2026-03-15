"""
Structured Logging
==================
Configures structured JSON logging for production observability.
In development, outputs human-readable colored logs.
In production, outputs JSON for log aggregation (Datadog, CloudWatch, etc.)
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

    # Shared processors for both dev and prod
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.ExceptionRenderer(),
    ]

    if settings.is_development:
        # Human-readable output for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]
    else:
        # JSON output for production log aggregation
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
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging to use structlog
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