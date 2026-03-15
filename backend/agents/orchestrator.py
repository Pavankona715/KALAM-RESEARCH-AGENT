"""
Agent Factory
=============
Creates agent instances with proper dependency injection.
Caches instances to avoid rebuilding LangGraph graphs on every request.

Usage:
    factory = get_agent_factory()
    agent = factory.get_agent("react", llm=router)
    result = await agent.run(user_message="...", session_id="...", user_id="...")
"""

from __future__ import annotations

from typing import Optional

from backend.agents.react_agent import ReActAgent
from backend.agents.base import BaseAgent
from backend.llm.router import LLMRouter, get_llm_router
from backend.tools.registry import ToolRegistry, get_tool_registry
from backend.observability.logger import get_logger

logger = get_logger(__name__)

# Map agent_type strings to agent classes
AGENT_REGISTRY: dict[str, type[BaseAgent]] = {
    "react": ReActAgent,
    # "planner": PlannerAgent,     # Added in Step 10
    # "researcher": ResearchAgent, # Added in Step 10
    # "analyst": AnalystAgent,     # Added in Step 10
    # "writer": WriterAgent,       # Added in Step 10
}


class AgentFactory:
    """
    Creates and caches agent instances.

    Caching matters because LangGraph compiles the graph on first call —
    this involves building the state machine, validating edges, etc.
    We don't want to pay that cost on every HTTP request.
    """

    def __init__(self, llm: LLMRouter, tool_registry: ToolRegistry):
        self._llm = llm
        self._tool_registry = tool_registry
        self._cache: dict[str, BaseAgent] = {}

    def get_agent(
        self,
        agent_type: str,
        max_steps: int = 25,
    ) -> BaseAgent:
        """
        Get a cached agent instance or create a new one.

        Args:
            agent_type: One of "react", "planner", "researcher", etc.
            max_steps: Maximum reasoning cycles (affects cache key)
        """
        cache_key = f"{agent_type}:{max_steps}"

        if cache_key not in self._cache:
            agent_class = AGENT_REGISTRY.get(agent_type)
            if not agent_class:
                raise ValueError(
                    f"Unknown agent type: '{agent_type}'. "
                    f"Available: {list(AGENT_REGISTRY.keys())}"
                )

            agent = agent_class(
                llm=self._llm,
                tool_registry=self._tool_registry,
                max_steps=max_steps,
            )
            self._cache[cache_key] = agent
            logger.info("agent_created", agent_type=agent_type, max_steps=max_steps)

        return self._cache[cache_key]

    def clear_cache(self) -> None:
        """Clear all cached agents (useful for testing)."""
        self._cache.clear()


# ─── Singleton ────────────────────────────────────────────────────────────────

_factory: Optional[AgentFactory] = None


def get_agent_factory() -> AgentFactory:
    """Get or create the global agent factory."""
    global _factory
    if _factory is None:
        _factory = AgentFactory(
            llm=get_llm_router(),
            tool_registry=get_tool_registry(),
        )
    return _factory