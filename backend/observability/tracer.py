"""
Distributed Tracing
===================
Configures LangSmith for LLM/agent tracing and OpenTelemetry for system tracing.

LangSmith: Captures LLM calls, agent reasoning steps, tool usage, token counts.
OpenTelemetry: Captures HTTP requests, DB queries, service dependencies.
"""

import os
from typing import Optional

from backend.config.settings import get_settings
from backend.observability.logger import get_logger

logger = get_logger(__name__)


def configure_langsmith() -> bool:
    """
    Configure LangSmith tracing for LangChain/LangGraph.
    Returns True if successfully configured, False otherwise.
    """
    settings = get_settings()

    if not settings.langchain_api_key:
        logger.warning("langsmith_disabled", reason="LANGCHAIN_API_KEY not configured")
        return False

    if not settings.langchain_tracing_v2:
        logger.info("langsmith_disabled", reason="LANGCHAIN_TRACING_V2=false")
        return False

    # LangSmith reads these from environment
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint

    logger.info(
        "langsmith_configured",
        project=settings.langchain_project,
        endpoint=settings.langchain_endpoint,
    )
    return True


def configure_tracing() -> None:
    """
    Initialize all observability tooling.
    Call once at application startup.
    """
    langsmith_enabled = configure_langsmith()

    logger.info(
        "observability_configured",
        langsmith=langsmith_enabled,
    )