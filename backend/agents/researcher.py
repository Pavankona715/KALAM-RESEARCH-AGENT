"""
Researcher Agent
================
Second agent in the multi-agent pipeline.

Responsibility: Execute the research subtasks from the planner's plan.
Uses web search, Wikipedia, and the RAG knowledge base to gather
factual, up-to-date information.

The researcher collects raw information — it does NOT analyze or
draw conclusions. That's the analyst's job.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import AIMessage

from backend.agents.multi_agent_state import MultiAgentState
from backend.config.prompts import RESEARCH_AGENT_SYSTEM_PROMPT
from backend.llm.base import LLMRequest, Message, ToolDefinition
from backend.llm.router import LLMRouter
from backend.tools.registry import ToolRegistry
from backend.observability.logger import get_logger

logger = get_logger(__name__)

RESEARCHER_TOOLS = ["web_search", "wikipedia", "file_reader"]
MAX_RESEARCH_STEPS = 5


class ResearchAgent:
    """
    Gathers information from multiple sources using tools.

    Runs a mini ReAct loop — thinks about what to search,
    executes searches, processes results, repeats until
    enough information is collected.
    """

    def __init__(self, llm: LLMRouter, tool_registry: ToolRegistry):
        self.llm = llm
        self.tool_registry = tool_registry

    async def run(self, state: MultiAgentState) -> dict:
        """
        Research all subtasks assigned to the researcher.
        Returns accumulated findings and sources.
        """
        logger.info(
            "researcher_started",
            run_id=state.get("run_id"),
        )

        plan = state.get("plan") or {}
        research_tasks = [
            t for t in plan.get("subtasks", [])
            if t.get("agent") == "researcher"
        ]

        if not research_tasks:
            # No research tasks — use the original request
            research_tasks = [{"description": state["user_request"]}]

        # Get available tools
        tools = self.tool_registry.get_tools(RESEARCHER_TOOLS)
        tool_definitions = [
            ToolDefinition(
                name=t.name,
                description=t.description,
                parameters=t.parameters,
            )
            for t in tools
        ]

        # Also include any retrieved RAG context
        rag_context = ""
        retrieved_docs = state.get("metadata", {}).get("retrieved_docs", [])
        if retrieved_docs:
            doc_texts = [
                f"[Doc: {d.get('source', 'unknown')}]\n{d.get('content', '')[:500]}"
                for d in retrieved_docs[:3]
            ]
            rag_context = "\n\n".join(doc_texts)

        # Build research prompt incorporating the plan
        task_descriptions = "\n".join(
            f"- {t.get('description', '')}" for t in research_tasks
        )

        system_content = RESEARCH_AGENT_SYSTEM_PROMPT
        if rag_context:
            system_content += f"\n\n## Knowledge Base Context\n{rag_context}"

        messages = [
            Message(role="system", content=system_content),
            Message(
                role="user",
                content=(
                    f"Research objective: {plan.get('objective', state['user_request'])}\n\n"
                    f"Research tasks:\n{task_descriptions}\n\n"
                    "Use the available tools to gather comprehensive, accurate information. "
                    "Cite your sources."
                ),
            ),
        ]

        findings_parts = []
        sources = []
        step = 0

        # Mini ReAct loop for research
        while step < MAX_RESEARCH_STEPS:
            step += 1

            response = await self.llm.complete(LLMRequest(
                messages=messages,
                tools=tool_definitions if tool_definitions else None,
                metadata={"agent": "researcher", "run_id": state.get("run_id", "")},
            ))

            if response.content:
                findings_parts.append(response.content)

            if not response.has_tool_calls:
                break

            # Execute tool calls
            tool_results = await asyncio.gather(*[
                self.tool_registry.execute(tc.name, tc.arguments)
                for tc in response.tool_calls
            ])

            # Add tool results to messages for next iteration
            from langchain_core.messages import AIMessage as LCAIMessage, ToolMessage
            for tc, result in zip(response.tool_calls, tool_results):
                tool_content = result.to_llm_string()
                messages.append(Message(
                    role="tool",
                    content=tool_content,
                    tool_call_id=tc.id,
                ))
                if result.success and result.output:
                    sources.append(f"{tc.name}: {result.output[:100]}...")

            # Add assistant message showing tool calls were made
            messages.append(Message(
                role="assistant",
                content=response.content or f"[Used tools: {[tc.name for tc in response.tool_calls]}]",
            ))

        findings = "\n\n".join(findings_parts) or "No specific findings gathered."

        logger.info(
            "researcher_completed",
            run_id=state.get("run_id"),
            steps=step,
            sources_found=len(sources),
        )

        return {
            "research_findings": findings,
            "sources": sources,
            "current_agent": "analyst",
            "completed_agents": state.get("completed_agents", []) + ["researcher"],
            "messages": [AIMessage(
                content=f"**Research complete** ({step} steps, {len(sources)} sources)\n\n"
                        f"{findings[:300]}..."
                if len(findings) > 300 else f"**Research complete**\n\n{findings}"
            )],
        }