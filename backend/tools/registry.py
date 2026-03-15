"""
Tool Registry
=============
Central registry for all available tools.

Responsibilities:
1. Register tools (manually or by auto-discovery)
2. Look up tools by name
3. Enforce permission sets — an agent only gets tools it's been granted
4. Convert tool list to LLM function definitions

Design: The registry is a singleton. Tools register on import.
Auto-discovery imports all modules in the tools/ package, triggering
registration via module-level instantiation.

Usage:
    # Register a tool
    registry = get_tool_registry()
    registry.register(WebSearchTool())

    # Get tools for an agent
    tools = registry.get_tools(["web_search", "calculator"])

    # Get all tool definitions for LLM
    definitions = registry.to_llm_definitions(["web_search"])
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Optional

from backend.tools.base import BaseTool, ToolResult
from backend.observability.logger import get_logger

logger = get_logger(__name__)


class ToolRegistry:
    """
    Manages tool registration and lookup.

    The registry is intentionally simple — it's just a dict with
    permission-checking logic on top.
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance. Overwrites if name already registered."""
        if tool.name in self._tools:
            logger.warning("tool_overwritten", tool_name=tool.name)
        self._tools[tool.name] = tool
        logger.debug("tool_registered", tool_name=tool.name)

    def get(self, name: str) -> Optional[BaseTool]:
        """Look up a tool by name. Returns None if not found."""
        return self._tools.get(name)

    def get_tools(self, names: list[str]) -> list[BaseTool]:
        """
        Get multiple tools by name.
        Logs warnings for any names not found (silent fail — don't break agents).
        """
        tools = []
        for name in names:
            tool = self._tools.get(name)
            if tool:
                tools.append(tool)
            else:
                logger.warning("tool_not_found", tool_name=name)
        return tools

    def all_tools(self) -> list[BaseTool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def all_names(self) -> list[str]:
        """Return names of all registered tools."""
        return list(self._tools.keys())

    def to_llm_definitions(self, names: Optional[list[str]] = None) -> list[dict]:
        """
        Convert tools to LLM function definitions.
        Pass names to filter, or None to include all.
        """
        tools = self.get_tools(names) if names else self.all_tools()
        return [t.to_llm_definition() for t in tools]

    async def execute(self, tool_name: str, tool_input: dict) -> ToolResult:
        """
        Execute a tool by name. Returns error ToolResult if tool not found.
        """
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"Tool '{tool_name}' not found in registry. "
                      f"Available: {self.all_names()}",
            )
        return await tool.run(tool_input)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# ─── Permission Sets ──────────────────────────────────────────────────────────

# Pre-defined tool permission sets for different agent types.
# Agents should only get the tools they need — principle of least privilege.
TOOL_PERMISSIONS: dict[str, list[str]] = {
    "react": [
        "web_search",
        "calculator",
        "wikipedia",
        "file_reader",
    ],
    "researcher": [
        "web_search",
        "wikipedia",
        "file_reader",
        "document_loader",
    ],
    "analyst": [
        "calculator",
        "database_query",
        "file_reader",
    ],
    "writer": [
        "file_reader",
    ],
    "admin": [
        "web_search",
        "calculator",
        "wikipedia",
        "file_reader",
        "document_loader",
        "database_query",
    ],
}


def get_tools_for_agent(agent_type: str) -> list[str]:
    """Get the allowed tool names for a given agent type."""
    return TOOL_PERMISSIONS.get(agent_type, TOOL_PERMISSIONS["react"])


# ─── Singleton + Auto-discovery ───────────────────────────────────────────────

_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """
    Get the global tool registry, auto-discovering all tools on first call.
    """
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
        _auto_discover_tools(_registry)
    return _registry


def _auto_discover_tools(registry: ToolRegistry) -> None:
    """
    Import all modules in backend/tools/ to trigger tool registration.
    Tools register themselves at module level when imported.
    """
    import backend.tools as tools_pkg

    # Skip these — they're framework files, not tools
    SKIP_MODULES = {"base", "registry", "__init__"}

    for module_info in pkgutil.iter_modules(tools_pkg.__path__):
        if module_info.name in SKIP_MODULES:
            continue
        try:
            module = importlib.import_module(f"backend.tools.{module_info.name}")
            # Tools register themselves when the module loads via module-level code
            logger.debug("tool_module_loaded", module=module_info.name)
        except ImportError as e:
            # Missing optional dependency (e.g. tavily not installed)
            logger.warning(
                "tool_module_skipped",
                module=module_info.name,
                reason=str(e),
            )