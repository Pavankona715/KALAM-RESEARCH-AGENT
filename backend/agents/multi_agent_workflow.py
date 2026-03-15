"""
Multi-Agent Orchestrator
========================
LangGraph workflow that connects Planner → Researcher → Analyst → Writer.

The workflow is a simple linear graph — each agent runs sequentially.
Future extension: add conditional edges to skip agents or loop back
for clarification based on confidence scores.

Usage:
    workflow = MultiAgentWorkflow(llm=router, tool_registry=registry)
    result = await workflow.run(
        user_request="Write a report on the state of AI in 2024",
        session_id="sess-123",
        user_id="user-456",
    )
    print(result["final_report"])
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from backend.agents.analyst import AnalystAgent
from backend.agents.multi_agent_state import MultiAgentState, create_multi_agent_state
from backend.agents.planner import PlannerAgent
from backend.agents.researcher import ResearchAgent
from backend.agents.writer import WriterAgent
from backend.llm.router import LLMRouter, get_llm_router
from backend.tools.registry import ToolRegistry, get_tool_registry
from backend.observability.logger import get_logger

logger = get_logger(__name__)


class MultiAgentWorkflow:
    """
    Orchestrates the full Planner → Researcher → Analyst → Writer pipeline.

    Each agent is a LangGraph node. The state flows through all nodes
    sequentially, with each agent enriching the shared state.
    """

    def __init__(
        self,
        llm: Optional[LLMRouter] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        self.llm = llm or get_llm_router()
        self.tool_registry = tool_registry or get_tool_registry()

        # Instantiate individual agents
        self.planner = PlannerAgent(llm=self.llm)
        self.researcher = ResearchAgent(llm=self.llm, tool_registry=self.tool_registry)
        self.analyst = AnalystAgent(llm=self.llm)
        self.writer = WriterAgent(llm=self.llm)

        # Compile the graph once
        self._graph = self._build_graph()

    def _build_graph(self):
        """Build the sequential multi-agent LangGraph."""
        graph = StateGraph(MultiAgentState)

        # Register nodes — each wraps an agent's run() method
        graph.add_node("planner", self.planner.run)
        graph.add_node("researcher", self.researcher.run)
        graph.add_node("analyst", self.analyst.run)
        graph.add_node("writer", self.writer.run)

        # Linear sequential edges
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "researcher")
        graph.add_edge("researcher", "analyst")
        graph.add_edge("analyst", "writer")
        graph.add_edge("writer", END)

        return graph.compile()

    async def run(
        self,
        user_request: str,
        session_id: str,
        user_id: str,
        run_id: Optional[str] = None,
        retrieved_docs: Optional[list] = None,
    ) -> MultiAgentState:
        """
        Run the full multi-agent pipeline to completion.

        Args:
            user_request: The user's research question or task
            session_id: Current chat session ID
            user_id: User ID for permission checks
            run_id: Optional tracking ID
            retrieved_docs: Pre-retrieved RAG documents to give context

        Returns:
            Final MultiAgentState with final_report populated.
        """
        if run_id is None:
            run_id = str(uuid.uuid4())

        initial_state = create_multi_agent_state(
            user_request=user_request,
            session_id=session_id,
            user_id=user_id,
            run_id=run_id,
        )

        # Pass retrieved docs through metadata for researcher access
        if retrieved_docs:
            serialized = []
            for doc in retrieved_docs:
                if hasattr(doc, "content"):
                    serialized.append({
                        "content": doc.content,
                        "source": doc.source,
                        "score": doc.score,
                    })
                elif isinstance(doc, dict):
                    serialized.append(doc)
            initial_state["metadata"]["retrieved_docs"] = serialized

        start = time.perf_counter()
        logger.info(
            "multi_agent_workflow_started",
            run_id=run_id,
            request=user_request[:100],
        )

        try:
            final_state = await self._graph.ainvoke(
                initial_state,
                config={"recursion_limit": 20},
            )

            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "multi_agent_workflow_completed",
                run_id=run_id,
                completed_agents=final_state.get("completed_agents", []),
                report_length=len(final_state.get("final_report", "")),
                latency_ms=round(elapsed_ms, 2),
            )

            return final_state

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "multi_agent_workflow_failed",
                run_id=run_id,
                error=str(e),
                latency_ms=round(elapsed_ms, 2),
            )
            raise


# Update the orchestrator factory to include multi_agent
def get_multi_agent_workflow(
    llm: Optional[LLMRouter] = None,
    tool_registry: Optional[ToolRegistry] = None,
) -> MultiAgentWorkflow:
    """Create a MultiAgentWorkflow instance with default dependencies."""
    return MultiAgentWorkflow(llm=llm, tool_registry=tool_registry)