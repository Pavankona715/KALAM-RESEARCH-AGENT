"""
Planner Agent
=============
First agent in the multi-agent pipeline.

Responsibility: Analyze the user's request and produce a structured plan
that tells the other agents exactly what to do.

The planner does NOT do any research or writing — it only plans.
This separation ensures the plan is thoughtful and well-structured
before any expensive work begins.

Output format (JSON):
{
    "objective": "Clear statement of what we're trying to accomplish",
    "subtasks": [
        {
            "id": "task_1",
            "description": "Search for X",
            "agent": "researcher",
            "expected_output": "List of sources about X"
        }
    ],
    "output_format": "report",  # report | answer | analysis | summary
    "estimated_complexity": "medium"  # low | medium | high
}
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage

from backend.agents.multi_agent_state import MultiAgentState
from backend.config.prompts import PLANNER_AGENT_SYSTEM_PROMPT
from backend.llm.base import LLMRequest, Message
from backend.llm.router import LLMRouter
from backend.observability.logger import get_logger

logger = get_logger(__name__)


class PlannerAgent:
    """
    Creates execution plans for complex research requests.

    Called first in the multi-agent workflow.
    Its output (the plan) guides every subsequent agent.
    """

    def __init__(self, llm: LLMRouter):
        self.llm = llm

    async def run(self, state: MultiAgentState) -> dict:
        """
        Generate a structured execution plan for the user's request.
        Returns partial state update with the plan.
        """
        logger.info(
            "planner_started",
            run_id=state.get("run_id"),
            request=state["user_request"][:100],
        )

        system_prompt = PLANNER_AGENT_SYSTEM_PROMPT

        messages = [
            Message(role="system", content=system_prompt),
            Message(
                role="user",
                content=(
                    f"Create a research plan for:\n\n{state['user_request']}\n\n"
                    "Respond with valid JSON only. No markdown, no explanation."
                ),
            ),
        ]

        try:
            response = await self.llm.complete(LLMRequest(
                messages=messages,
                response_format={"type": "json_object"},
                metadata={
                    "agent": "planner",
                    "run_id": state.get("run_id", ""),
                },
            ))

            plan = self._parse_plan(response.content, state["user_request"])

        except Exception as e:
            logger.error("planner_failed", error=str(e))
            # Fallback: create a simple default plan
            plan = self._default_plan(state["user_request"])

        logger.info(
            "planner_completed",
            run_id=state.get("run_id"),
            subtask_count=len(plan.get("subtasks", [])),
        )

        return {
            "plan": plan,
            "current_agent": "researcher",
            "completed_agents": state.get("completed_agents", []) + ["planner"],
            "messages": [AIMessage(
                content=f"**Plan created:** {plan.get('objective', 'Research task')}\n"
                        f"Subtasks: {len(plan.get('subtasks', []))} | "
                        f"Format: {plan.get('output_format', 'report')}"
            )],
        }

    def _parse_plan(self, content: str, user_request: str) -> dict:
        """Parse LLM JSON response into a plan dict."""
        try:
            plan = json.loads(content)
            # Validate required fields
            if "objective" not in plan:
                plan["objective"] = user_request
            if "subtasks" not in plan or not isinstance(plan["subtasks"], list):
                plan["subtasks"] = self._default_subtasks(user_request)
            if "output_format" not in plan:
                plan["output_format"] = "report"
            return plan
        except (json.JSONDecodeError, Exception):
            return self._default_plan(user_request)

    def _default_plan(self, user_request: str) -> dict:
        """Fallback plan when LLM fails or returns invalid JSON."""
        return {
            "objective": user_request,
            "subtasks": self._default_subtasks(user_request),
            "output_format": "report",
            "estimated_complexity": "medium",
        }

    def _default_subtasks(self, user_request: str) -> list[dict]:
        return [
            {
                "id": "task_1",
                "description": f"Research: {user_request}",
                "agent": "researcher",
                "expected_output": "Key findings and sources",
            },
            {
                "id": "task_2",
                "description": "Analyze the research findings",
                "agent": "analyst",
                "expected_output": "Structured analysis with insights",
            },
        ]