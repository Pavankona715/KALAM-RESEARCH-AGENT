"""
Multi-Agent Workflow State
==========================
Extends the base AgentState with fields specific to multi-agent coordination.

Each agent writes its output into the shared state.
Subsequent agents read prior agents' outputs when reasoning.

State flow:
  user_request
    → plan (from PlannerAgent)
    → research_findings (from ResearchAgent)
    → analysis (from AnalystAgent)
    → final_report (from WriterAgent)
"""

from __future__ import annotations

from typing import Annotated, Any, Optional

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class MultiAgentState(TypedDict):
    """
    Shared state for the multi-agent workflow.

    All agents read from and write to this state.
    Fields are populated progressively as agents complete.
    """

    # Core conversation
    messages: Annotated[list[BaseMessage], add_messages]
    user_request: str
    session_id: str
    user_id: str

    # Planner output
    plan: Optional[dict]
    # {
    #   "objective": str,
    #   "subtasks": [{"id": str, "description": str, "agent": str}],
    #   "output_format": str,  # "report" | "answer" | "analysis"
    # }

    # Researcher output
    research_findings: Optional[str]
    sources: list[str]

    # Analyst output
    analysis: Optional[str]
    key_insights: list[str]

    # Writer output
    final_report: Optional[str]

    # Execution tracking
    current_agent: str
    completed_agents: list[str]
    errors: list[str]

    # Run metadata
    run_id: Optional[str]
    metadata: dict[str, Any]


def create_multi_agent_state(
    user_request: str,
    session_id: str,
    user_id: str,
    run_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> MultiAgentState:
    """Create initial state for a multi-agent run."""
    from langchain_core.messages import HumanMessage

    return MultiAgentState(
        messages=[HumanMessage(content=user_request)],
        user_request=user_request,
        session_id=session_id,
        user_id=user_id,
        plan=None,
        research_findings=None,
        sources=[],
        analysis=None,
        key_insights=[],
        final_report=None,
        current_agent="planner",
        completed_agents=[],
        errors=[],
        run_id=run_id,
        metadata=metadata or {},
    )