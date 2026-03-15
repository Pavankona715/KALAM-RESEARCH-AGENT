"""
LLM Provider Interface
======================
Defines the contract every LLM provider adapter must fulfill.

Uses Python Protocol (structural typing) instead of ABC:
- No inheritance required — any class with the right methods qualifies
- Easier to mock in tests (just implement the protocol)
- mypy validates compliance statically

Data flow:
    LLMRequest → Provider.complete() → LLMResponse
    LLMRequest → Provider.stream()   → AsyncGenerator[StreamChunk]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Optional, Protocol, runtime_checkable


# ─── Token Usage ──────────────────────────────────────────────────────────────

@dataclass
class TokenUsage:
    """Token counts from an LLM API call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Allows accumulating usage across multiple calls."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


# ─── Tool Calling ─────────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    """
    A tool invocation requested by the LLM.
    Returned in LLMResponse when the model decides to use a tool.
    """
    id: str                    # Tool call ID from the provider (for multi-turn)
    name: str                  # Tool/function name
    arguments: dict[str, Any]  # Parsed JSON arguments


@dataclass
class ToolDefinition:
    """
    Describes a tool to the LLM.
    Passed in LLMRequest.tools to tell the model what it can call.
    """
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the tool's parameters

    def to_openai_format(self) -> dict:
        """Convert to OpenAI/LiteLLM function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


# ─── Messages ─────────────────────────────────────────────────────────────────

@dataclass
class Message:
    """A single message in the conversation."""
    role: str   # system | user | assistant | tool
    content: str
    tool_call_id: Optional[str] = None   # Required for role="tool" messages
    tool_calls: Optional[list[ToolCall]] = None  # Set on assistant messages

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": str(tc.arguments)},
                }
                for tc in self.tool_calls
            ]
        return d


# ─── Request ──────────────────────────────────────────────────────────────────

@dataclass
class LLMRequest:
    """
    Everything needed to make an LLM API call.
    Provider-agnostic — LiteLLM translates this to the specific provider format.
    """
    messages: list[Message]

    # Model config
    model: Optional[str] = None         # Overrides default if set
    temperature: Optional[float] = None  # Overrides default if set
    max_tokens: Optional[int] = None

    # Tool calling
    tools: Optional[list[ToolDefinition]] = None
    tool_choice: str = "auto"  # auto | none | required

    # Structured output
    response_format: Optional[dict] = None  # {"type": "json_object"}

    # LiteLLM extras
    metadata: dict[str, Any] = field(default_factory=dict)
    # Used for LangSmith tagging: {"session_id": "...", "agent": "researcher"}


# ─── Response ─────────────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    """
    Normalized response from any LLM provider.
    Always has content OR tool_calls (or both), never neither.
    """
    content: str                              # Text response (may be empty if tool_calls set)
    usage: TokenUsage
    model: str                                # Actual model used (after routing/fallback)
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"               # stop | tool_calls | length | content_filter
    latency_ms: float = 0.0
    raw: Optional[dict] = None                # Full provider response for debugging

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def is_complete(self) -> bool:
        return self.finish_reason in ("stop", "tool_calls")


@dataclass
class StreamChunk:
    """A single chunk from a streaming LLM response."""
    content: str = ""
    tool_call_delta: Optional[dict] = None
    finish_reason: Optional[str] = None
    usage: Optional[TokenUsage] = None   # Only in the final chunk


# ─── Provider Protocol ────────────────────────────────────────────────────────

@runtime_checkable
class LLMProvider(Protocol):
    """
    The contract every LLM provider must fulfill.

    @runtime_checkable allows isinstance() checks:
        assert isinstance(my_provider, LLMProvider)
    """

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Make a single LLM call and return the full response.
        Blocks until the model finishes generating.
        """
        ...

    async def stream(
        self, request: LLMRequest
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream the LLM response token by token.
        Yields StreamChunk objects. Last chunk has finish_reason set.
        """
        ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.
        Returns list of embedding vectors (one per input text).
        """
        ...

    def get_model_name(self) -> str:
        """Return the currently configured model name."""
        ...


# ─── Exceptions ───────────────────────────────────────────────────────────────

class LLMError(Exception):
    """Base exception for all LLM provider errors."""
    def __init__(self, message: str, provider: str = "", model: str = ""):
        super().__init__(message)
        self.provider = provider
        self.model = model


class LLMRateLimitError(LLMError):
    """Rate limit hit. Retry after backoff."""
    pass


class LLMContextLengthError(LLMError):
    """Input too long for model's context window."""
    pass


class LLMAuthError(LLMError):
    """Invalid API key or insufficient permissions."""
    pass


class LLMProviderUnavailableError(LLMError):
    """Provider is down or unreachable."""
    pass