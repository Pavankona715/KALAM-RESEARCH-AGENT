"""
Writer Agent
============
Final agent in the multi-agent pipeline.

Responsibility: Take all accumulated context (research + analysis)
and produce a polished, well-structured final output.

The output format depends on the plan's output_format field:
  "report"   → Full structured report with sections
  "answer"   → Concise direct answer
  "analysis" → Analytical document
  "summary"  → Executive summary
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from backend.agents.multi_agent_state import MultiAgentState
from backend.config.prompts import WRITER_AGENT_SYSTEM_PROMPT
from backend.llm.base import LLMRequest, Message
from backend.llm.router import LLMRouter
from backend.observability.logger import get_logger

logger = get_logger(__name__)

FORMAT_INSTRUCTIONS = {
    "report": (
        "Write a comprehensive report with: Executive Summary, Background, "
        "Key Findings, Analysis, Conclusions, and Sources."
    ),
    "answer": (
        "Write a clear, direct answer. Be concise but complete. "
        "Support your answer with evidence from the research."
    ),
    "analysis": (
        "Write a structured analysis with: Overview, Detailed Analysis, "
        "Key Insights, Implications, and Recommendations."
    ),
    "summary": (
        "Write an executive summary: What was asked, what was found, "
        "key takeaways, and recommended next steps. Maximum 3 paragraphs."
    ),
}


class WriterAgent:
    """
    Produces the final polished output from accumulated research and analysis.

    Input: research_findings + analysis + key_insights from prior agents
    Output: final_report (the response returned to the user)
    """

    def __init__(self, llm: LLMRouter):
        self.llm = llm

    async def run(self, state: MultiAgentState) -> dict:
        """
        Write the final output based on all accumulated context.
        """
        logger.info(
            "writer_started",
            run_id=state.get("run_id"),
        )

        plan = state.get("plan") or {}
        output_format = plan.get("output_format", "report")
        format_instruction = FORMAT_INSTRUCTIONS.get(output_format, FORMAT_INSTRUCTIONS["report"])

        # Build context from all prior agents
        context_parts = [
            f"**Original Request:** {state['user_request']}",
            f"**Objective:** {plan.get('objective', state['user_request'])}",
        ]

        if state.get("research_findings"):
            context_parts.append(
                f"**Research Findings:**\n{state['research_findings']}"
            )

        if state.get("analysis"):
            context_parts.append(
                f"**Analysis:**\n{state['analysis']}"
            )

        if state.get("key_insights"):
            insights_text = "\n".join(
                f"- {insight}" for insight in state["key_insights"]
            )
            context_parts.append(f"**Key Insights:**\n{insights_text}")

        if state.get("sources"):
            sources_text = "\n".join(state["sources"][:5])
            context_parts.append(f"**Sources:**\n{sources_text}")

        full_context = "\n\n".join(context_parts)

        messages = [
            Message(role="system", content=WRITER_AGENT_SYSTEM_PROMPT),
            Message(
                role="user",
                content=(
                    f"{full_context}\n\n"
                    f"**Output Format:** {output_format.upper()}\n"
                    f"{format_instruction}\n\n"
                    "Write the final output now."
                ),
            ),
        ]

        try:
            response = await self.llm.complete(LLMRequest(
                messages=messages,
                metadata={"agent": "writer", "run_id": state.get("run_id", "")},
            ))
            final_report = response.content

        except Exception as e:
            logger.error("writer_failed", error=str(e))
            # Fallback: return the analysis as the report
            final_report = state.get("analysis") or state.get("research_findings") or str(e)

        logger.info(
            "writer_completed",
            run_id=state.get("run_id"),
            report_length=len(final_report),
        )

        return {
            "final_report": final_report,
            "current_agent": "done",
            "completed_agents": state.get("completed_agents", []) + ["writer"],
            "messages": [AIMessage(content=final_report)],
        }