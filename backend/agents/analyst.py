"""
Analyst Agent
=============
Third agent in the multi-agent pipeline.

Responsibility: Take the researcher's raw findings and produce
structured analysis with clear insights, patterns, and conclusions.

The analyst does NOT gather new information — it works exclusively
with what the researcher found. It's a pure reasoning agent.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from backend.agents.multi_agent_state import MultiAgentState
from backend.config.prompts import ANALYST_AGENT_SYSTEM_PROMPT
from backend.llm.base import LLMRequest, Message
from backend.llm.router import LLMRouter
from backend.observability.logger import get_logger

logger = get_logger(__name__)


class AnalystAgent:
    """
    Analyzes research findings and extracts key insights.

    Pure reasoning — no tools needed.
    Input: research_findings from ResearchAgent
    Output: structured analysis + key_insights list
    """

    def __init__(self, llm: LLMRouter):
        self.llm = llm

    async def run(self, state: MultiAgentState) -> dict:
        """
        Analyze the research findings and produce structured insights.
        """
        logger.info(
            "analyst_started",
            run_id=state.get("run_id"),
        )

        research_findings = state.get("research_findings", "No research findings available.")
        plan = state.get("plan") or {}

        messages = [
            Message(role="system", content=ANALYST_AGENT_SYSTEM_PROMPT),
            Message(
                role="user",
                content=(
                    f"Objective: {plan.get('objective', state['user_request'])}\n\n"
                    f"Research Findings:\n{research_findings}\n\n"
                    "Analyze these findings. Provide:\n"
                    "1. A structured analysis\n"
                    "2. Key insights (as a numbered list)\n"
                    "3. Any gaps or limitations in the research\n"
                    "4. Confidence level in the findings"
                ),
            ),
        ]

        try:
            response = await self.llm.complete(LLMRequest(
                messages=messages,
                metadata={"agent": "analyst", "run_id": state.get("run_id", "")},
            ))
            analysis = response.content

        except Exception as e:
            logger.error("analyst_failed", error=str(e))
            analysis = f"Analysis unavailable due to error: {str(e)}"

        # Extract key insights (lines starting with numbers)
        key_insights = self._extract_insights(analysis)

        logger.info(
            "analyst_completed",
            run_id=state.get("run_id"),
            insights_found=len(key_insights),
        )

        return {
            "analysis": analysis,
            "key_insights": key_insights,
            "current_agent": "writer",
            "completed_agents": state.get("completed_agents", []) + ["analyst"],
            "messages": [AIMessage(
                content=f"**Analysis complete** ({len(key_insights)} key insights)\n\n"
                        f"{analysis[:300]}..."
                if len(analysis) > 300 else f"**Analysis complete**\n\n{analysis}"
            )],
        }

    def _extract_insights(self, analysis: str) -> list[str]:
        """Extract numbered insights from the analysis text."""
        insights = []
        for line in analysis.split("\n"):
            line = line.strip()
            # Match lines like "1. insight" or "- insight"
            if line and (
                (line[0].isdigit() and len(line) > 3 and line[1] in ".)")
                or line.startswith("- ")
                or line.startswith("• ")
            ):
                # Clean up the prefix
                text = line.lstrip("0123456789.-•) ").strip()
                if text and len(text) > 10:
                    insights.append(text)

        return insights[:10]  # Cap at 10 insights