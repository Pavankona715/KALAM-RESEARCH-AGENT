"""
LLM Layer
=========
Public API for the LLM abstraction layer.

Import from here, not from submodules:
    from backend.llm import LLMRequest, Message, get_llm_router
"""

from backend.llm.base import (
    LLMAuthError,
    LLMContextLengthError,
    LLMError,
    LLMProvider,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    Message,
    StreamChunk,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)
from backend.llm.litellm_provider import LiteLLMProvider
from backend.llm.router import LLMRouter, TaskComplexity, get_llm_router

__all__ = [
    "Message",
    "LLMRequest",
    "LLMResponse",
    "StreamChunk",
    "TokenUsage",
    "ToolCall",
    "ToolDefinition",
    "LLMProvider",
    "LiteLLMProvider",
    "LLMRouter",
    "TaskComplexity",
    "get_llm_router",
    "LLMError",
    "LLMRateLimitError",
    "LLMContextLengthError",
    "LLMAuthError",
    "LLMProviderUnavailableError",
]