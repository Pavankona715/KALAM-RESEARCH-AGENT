"""
Tool Base Classes
=================
Defines the contract every tool must fulfill.

Design:
- BaseTool is an abstract class (not Protocol) because tools share
  real implementation logic (validation, timing, error wrapping).
- Every tool gets input validation, execution timing, and error
  handling for free by inheriting from BaseTool.
- Tools declare their parameters as a JSON Schema dict — this gets
  passed directly to the LLM as a function definition.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolResult:
    """
    Standardized output from any tool execution.

    Agents receive ToolResult and decide what to do next based on
    success/failure and the output content.
    """
    tool_name: str
    success: bool
    output: str                           # Human-readable result for the LLM
    data: Optional[Any] = None            # Structured data for programmatic use
    error: Optional[str] = None           # Error message if success=False
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_llm_string(self) -> str:
        """
        Format result for injection into LLM context.
        Called by the agent to convert tool output into a message.
        """
        if self.success:
            return self.output
        return f"Tool '{self.tool_name}' failed: {self.error}"


class BaseTool(ABC):
    """
    Abstract base class for all tools.

    Subclasses must implement:
    - name: str                  unique tool identifier
    - description: str           tells the LLM what this tool does
    - parameters: dict           JSON Schema for tool inputs
    - _execute(input): str       the actual tool logic

    Subclasses get for free:
    - Input validation
    - Execution timing
    - Error wrapping into ToolResult
    - Consistent logging

    Usage:
        class MyTool(BaseTool):
            name = "my_tool"
            description = "Does something useful"
            parameters = {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Input query"}
                },
                "required": ["query"]
            }

            async def _execute(self, query: str) -> str:
                return f"Result for: {query}"
    """

    # Subclasses MUST define these as class attributes
    name: str
    description: str
    parameters: dict[str, Any]

    # Subclasses MAY override these
    timeout_seconds: float = 30.0
    max_output_length: int = 8000

    def __init_subclass__(cls, **kwargs):
        """Enforce that all required class attributes are defined."""
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "__abstractmethods__", None):
            for attr in ("name", "description", "parameters"):
                if not hasattr(cls, attr):
                    raise TypeError(
                        f"Tool class '{cls.__name__}' must define '{attr}'"
                    )

    @abstractmethod
    async def _execute(self, **kwargs: Any) -> str:
        """
        Core tool logic. Override this in subclasses.

        Args:
            **kwargs: Validated parameters matching the JSON Schema

        Returns:
            String output to be returned to the agent

        Raises:
            Any exception — will be caught and wrapped into ToolResult
        """
        ...

    async def run(self, tool_input: dict[str, Any]) -> ToolResult:
        """
        Public entry point. Validates input, executes, times, wraps result.
        Never raises — always returns a ToolResult (success or failure).
        """
        from backend.observability.logger import get_logger
        from backend.observability.metrics import ToolCallMetrics, metrics_collector
        logger = get_logger(self.name)

        start = time.perf_counter()

        # Validate required parameters
        validation_error = self._validate_input(tool_input)
        if validation_error:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"Invalid input: {validation_error}",
                latency_ms=0.0,
            )

        try:
            output = await self._execute(**tool_input)

            # Truncate if too long
            if len(output) > self.max_output_length:
                output = output[:self.max_output_length] + "\n[Output truncated]"

            latency_ms = (time.perf_counter() - start) * 1000

            metrics_collector.record_tool_call(ToolCallMetrics(
                tool_name=self.name,
                latency_ms=latency_ms,
                success=True,
            ))

            logger.debug("tool_executed", tool=self.name, latency_ms=round(latency_ms, 2))

            return ToolResult(
                tool_name=self.name,
                success=True,
                output=output,
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000

            metrics_collector.record_tool_call(ToolCallMetrics(
                tool_name=self.name,
                latency_ms=latency_ms,
                success=False,
                error=str(e),
            ))

            logger.error("tool_failed", tool=self.name, error=str(e))

            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=str(e),
                latency_ms=latency_ms,
            )

    def _validate_input(self, tool_input: dict[str, Any]) -> Optional[str]:
        """
        Validate that required parameters are present.
        Returns error message string if invalid, None if valid.
        """
        required = self.parameters.get("required", [])
        for param in required:
            if param not in tool_input:
                return f"Missing required parameter: '{param}'"
        return None

    def to_llm_definition(self) -> dict[str, Any]:
        """
        Format this tool as an LLM function definition.
        Compatible with OpenAI / LiteLLM tool calling format.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}'>"