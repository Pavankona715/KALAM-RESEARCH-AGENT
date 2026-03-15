"""
Base Agent
==========
Abstract base class for all agents in the system.

Every agent (ReAct, Planner, Researcher, etc.) shares:
- A reference to the LLM router
- A reference to the tool registry
- Standard run() and astream() interfaces
- Metric recording

Subclasses implement _build_graph() to define their LangGraph topology.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Optional

from backend.agents.state import AgentState
from backend.llm.router import LLMRouter
from backend.tools.registry import ToolRegistry
from backend.observability.logger import get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """
    Abstract base for all agents.

    Subclasses must implement:
        _build_graph() → CompiledGraph
        agent_type: str
    """

    agent_type: str = "base"

    def __init__(
        self,
        llm: LLMRouter,
        tool_registry: ToolRegistry,
        max_steps: int = 10,
    ):
        self.llm = llm
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self._graph = None  # Lazily compiled

    @property
    def graph(self):
        """Lazily build and cache the LangGraph compiled graph."""
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    @abstractmethod
    def _build_graph(self):
        """Build and return the compiled LangGraph graph."""
        ...

    async def run(
        self,
        user_message: str,
        session_id: str,
        user_id: str,
        conversation_history: Optional[list[dict]] = None,
        retrieved_docs: Optional[list] = None,
        run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentState:
        """
        Run the agent to completion and return the final state.

        Use this when you want the complete answer synchronously.
        For streaming intermediate steps, use astream() instead.
        """
        from backend.agents.state import create_initial_state

        if run_id is None:
            run_id = str(uuid.uuid4())

        initial_state = create_initial_state(
            user_message=user_message,
            session_id=session_id,
            user_id=user_id,
            agent_type=self.agent_type,
            max_steps=self.max_steps,
            conversation_history=conversation_history,
            retrieved_docs=retrieved_docs,
            run_id=run_id,
        )

        start = time.perf_counter()
        logger.info(
            "agent_run_started",
            agent_type=self.agent_type,
            run_id=run_id,
            session_id=session_id,
        )

        try:
            final_state = await self.graph.ainvoke(
                initial_state,
                config={"recursion_limit": self.max_steps + 5},
            )

            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "agent_run_completed",
                agent_type=self.agent_type,
                run_id=run_id,
                steps=final_state.get("step_count", 0),
                latency_ms=round(elapsed_ms, 2),
            )

            return final_state

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "agent_run_failed",
                agent_type=self.agent_type,
                run_id=run_id,
                error=str(e),
                latency_ms=round(elapsed_ms, 2),
            )
            raise

    async def astream(
        self,
        user_message: str,
        session_id: str,
        user_id: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Stream agent execution step by step.

        Yields state updates as the agent reasons and uses tools.
        Use this for real-time UI updates.

        Yields dicts with keys:
            type: "step" | "tool_call" | "tool_result" | "final"
            content: str
            metadata: dict
        """
        from backend.agents.state import create_initial_state

        initial_state = create_initial_state(
            user_message=user_message,
            session_id=session_id,
            user_id=user_id,
            agent_type=self.agent_type,
            max_steps=self.max_steps,
            conversation_history=conversation_history,
        )

        async for event in self.graph.astream(
            initial_state,
            config={"recursion_limit": self.max_steps + 5},
        ):
            # event is a dict of {node_name: state_update}
            for node_name, state_update in event.items():
                yield {
                    "type": "step",
                    "node": node_name,
                    "content": self._extract_content(state_update),
                    "metadata": {
                        "step": state_update.get("step_count", 0),
                    },
                }

    def _extract_content(self, state_update: dict) -> str:
        """Extract human-readable content from a state update."""
        messages = state_update.get("messages", [])
        if messages:
            last = messages[-1]
            if hasattr(last, "content"):
                return str(last.content)
        if state_update.get("final_answer"):
            return state_update["final_answer"]
        return ""