"""
ReAct Agent
===========
Implements the ReAct (Reasoning + Acting) pattern using LangGraph.

The agent alternates between:
  Reasoning: LLM analyzes the situation and decides what to do
  Acting:    Execute the chosen tool(s)
  Observing: Process tool results and reason about next steps

Graph topology:
  START → reasoning_node → [tool_node → reasoning_node]* → END

The routing function decides whether to loop (tool call needed)
or exit (final answer ready / max steps reached).

Key design choices:
- Tool calls happen in PARALLEL within a single step
- State carries full message history so LLM has complete context
- Hard max_steps limit prevents infinite loops
- Errors in individual tools don't abort the agent — it reasons about failures
"""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from langgraph.graph import END, START, StateGraph

from backend.agents.base import BaseAgent
from backend.agents.state import AgentState
from backend.config.prompts import REACT_AGENT_SYSTEM_PROMPT
from backend.llm.base import LLMRequest, Message, ToolDefinition
from backend.llm.router import LLMRouter
from backend.tools.registry import ToolRegistry, get_tools_for_agent
from backend.observability.logger import get_logger

logger = get_logger(__name__)


class ReActAgent(BaseAgent):
    """
    Single-agent ReAct loop.

    Uses the LLM to reason about which tools to call, executes them,
    then feeds results back to the LLM until a final answer is reached.

    Usage:
        agent = ReActAgent(llm=router, tool_registry=registry)
        result = await agent.run(
            user_message="What is the population of Tokyo?",
            session_id="session-123",
            user_id="user-456",
        )
        print(result["final_answer"])
    """

    agent_type = "react"

    def _build_graph(self):
        """Construct and compile the LangGraph state machine."""
        graph = StateGraph(AgentState)

        # Register nodes
        graph.add_node("reasoning", self._reasoning_node)
        graph.add_node("tools", self._tool_node)

        # Edges
        graph.add_edge(START, "reasoning")

        # Conditional edge: after reasoning, either use tools or finish
        graph.add_conditional_edges(
            "reasoning",
            self._should_use_tools,
            {
                "tools": "tools",    # LLM requested tool calls
                "end": END,          # LLM gave final answer
            },
        )

        # After tools run, always go back to reasoning
        graph.add_edge("tools", "reasoning")

        return graph.compile()

    # ─── Node: Reasoning ──────────────────────────────────────────────────────

    async def _reasoning_node(self, state: AgentState) -> dict:
        """
        Call the LLM to reason about next steps.

        The LLM can either:
        1. Request one or more tool calls (returns tool_calls in response)
        2. Produce a final answer (returns content with no tool calls)
        """
        step = state["step_count"] + 1

        # Safety: hard stop at max_steps
        if step > state["max_steps"]:
            logger.warning(
                "agent_max_steps_reached",
                run_id=state.get("run_id"),
                steps=step,
            )
            last_content = self._get_last_content(state)
            return {
                "step_count": step,
                "final_answer": (
                    last_content or
                    "I've reached the maximum number of reasoning steps. "
                    "Here is what I found so far: " +
                    self._summarize_tool_results(state)
                ),
                "messages": [
                    AIMessage(content="[Max steps reached — stopping]")
                ],
            }

        # Build tool definitions for this agent type
        allowed_tool_names = get_tools_for_agent(state["agent_type"])
        tools = self.tool_registry.get_tools(allowed_tool_names)
        tool_definitions = [
            ToolDefinition(
                name=t.name,
                description=t.description,
                parameters=t.parameters,
            )
            for t in tools
        ]

        # Build the system prompt
        system_prompt = self._build_system_prompt(state)

        # Convert LangChain messages to our LLM format
        history_messages = self._langchain_to_llm_messages(state["messages"])

        # Prepend system message
        all_messages = [Message(role="system", content=system_prompt)] + history_messages

        logger.debug(
            "reasoning_step",
            run_id=state.get("run_id"),
            step=step,
            message_count=len(all_messages),
            tools_available=len(tool_definitions),
        )

        # Call LLM
        llm_response = await self.llm.complete(LLMRequest(
            messages=all_messages,
            tools=tool_definitions if tool_definitions else None,
            metadata={
                "run_id": state.get("run_id", ""),
                "session_id": state["session_id"],
                "step": step,
            },
        ))

        # Build updates to return
        updates: dict[str, Any] = {"step_count": step}

        if llm_response.has_tool_calls:
            # LLM wants to use tools — convert to LangChain format
            tool_calls_for_lc = [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "args": tc.arguments,
                    "type": "tool_call",
                }
                for tc in llm_response.tool_calls
            ]

            ai_message = AIMessage(
                content=llm_response.content or "",
                tool_calls=tool_calls_for_lc,
            )
            updates["messages"] = [ai_message]
            updates["pending_tool_calls"] = [
                {"id": tc.id, "name": tc.name, "args": tc.arguments}
                for tc in llm_response.tool_calls
            ]

        else:
            # LLM gave a final answer
            ai_message = AIMessage(content=llm_response.content)
            updates["messages"] = [ai_message]
            updates["final_answer"] = llm_response.content
            updates["pending_tool_calls"] = []

        return updates

    # ─── Node: Tools ──────────────────────────────────────────────────────────

    async def _tool_node(self, state: AgentState) -> dict:
        """
        Execute all pending tool calls (in parallel where possible).
        Appends ToolMessage results to the message list.
        """
        import asyncio

        pending = state.get("pending_tool_calls", [])
        if not pending:
            return {"pending_tool_calls": [], "tool_results": []}

        logger.debug(
            "executing_tools",
            run_id=state.get("run_id"),
            tools=[t["name"] for t in pending],
        )

        # Execute all tool calls concurrently
        async def execute_one(tool_call: dict):
            result = await self.tool_registry.execute(
                tool_call["name"],
                tool_call.get("args", {}),
            )
            return tool_call["id"], tool_call["name"], result

        results = await asyncio.gather(
            *[execute_one(tc) for tc in pending],
            return_exceptions=True,
        )

        # Convert results to LangChain ToolMessages
        tool_messages = []
        tool_results = []

        for item in results:
            if isinstance(item, Exception):
                # Unexpected error executing the tool
                logger.error("tool_execution_error", error=str(item))
                continue

            call_id, tool_name, tool_result = item

            content = tool_result.to_llm_string()
            tool_messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id=call_id,
                    name=tool_name,
                )
            )
            tool_results.append({
                "tool": tool_name,
                "success": tool_result.success,
                "output": tool_result.output[:500],  # Truncate for state storage
            })

            logger.debug(
                "tool_executed",
                tool=tool_name,
                success=tool_result.success,
                run_id=state.get("run_id"),
            )

        return {
            "messages": tool_messages,
            "pending_tool_calls": [],
            "tool_results": state.get("tool_results", []) + tool_results,
        }

    # ─── Routing ──────────────────────────────────────────────────────────────

    def _should_use_tools(
        self, state: AgentState
    ) -> Literal["tools", "end"]:
        """
        Route to tool execution or END based on last message.
        """
        # If final_answer is set, we're done
        if state.get("final_answer"):
            return "end"

        # If there are pending tool calls, execute them
        if state.get("pending_tool_calls"):
            return "tools"

        # Check the last message for tool calls
        messages = state.get("messages", [])
        if messages:
            last = messages[-1]
            if isinstance(last, AIMessage) and last.tool_calls:
                return "tools"

        return "end"

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _build_system_prompt(self, state: AgentState) -> str:
        """Build the system prompt with current context summary."""
        from datetime import datetime, timezone
        tool_count = len(get_tools_for_agent(state["agent_type"]))

        context_summary = (
            f"- Step {state['step_count'] + 1} of {state['max_steps']} maximum\n"
            f"- {tool_count} tools available\n"
            f"- {len(state.get('tool_results', []))} tools used so far this session"
        )

        return REACT_AGENT_SYSTEM_PROMPT.format(
            context_summary=context_summary,
            current_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        )

    def _langchain_to_llm_messages(
        self, lc_messages: list
    ) -> list[Message]:
        """Convert LangChain message objects to our LLM Message dataclass."""
        messages = []
        for msg in lc_messages:
            if isinstance(msg, SystemMessage):
                messages.append(Message(role="system", content=str(msg.content)))
            elif isinstance(msg, AIMessage):
                content = str(msg.content) if msg.content else ""
                messages.append(Message(role="assistant", content=content))
            elif isinstance(msg, ToolMessage):
                messages.append(Message(
                    role="tool",
                    content=str(msg.content),
                    tool_call_id=msg.tool_call_id,
                ))
            else:
                # HumanMessage and others
                messages.append(Message(role="user", content=str(msg.content)))
        return messages

    def _get_last_content(self, state: AgentState) -> str:
        """Get the content of the last AI message."""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                return str(msg.content)
        return ""

    def _summarize_tool_results(self, state: AgentState) -> str:
        """Summarize tool results collected so far."""
        results = state.get("tool_results", [])
        if not results:
            return "No tool results collected."
        lines = []
        for r in results[-3:]:  # Last 3 results
            status = "✓" if r["success"] else "✗"
            lines.append(f"{status} {r['tool']}: {r['output'][:200]}")
        return "\n".join(lines)