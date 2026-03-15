"""
ReAct Agent Unit Tests
======================
Tests for AgentState, ReActAgent routing logic, and tool execution node.
The LLM is fully mocked — no real API calls.
"""

import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.agents.state import AgentState, create_initial_state
from backend.tools.base import BaseTool, ToolResult
from backend.tools.registry import ToolRegistry


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_state(**overrides) -> AgentState:
    """Build a minimal valid AgentState for testing."""
    base = AgentState(
        messages=[HumanMessage(content="What is 2+2?")],
        session_id="sess-123",
        user_id="user-456",
        agent_type="react",
        step_count=0,
        max_steps=5,
        pending_tool_calls=[],
        tool_results=[],
        final_answer=None,
        error=None,
        run_id="run-789",
        metadata={},
    )
    base.update(overrides)
    return base


def make_llm_response(content="", tool_calls=None):
    """Build a mock LLMResponse."""
    from backend.llm.base import LLMResponse, TokenUsage, ToolCall
    tcs = tool_calls or []
    return LLMResponse(
        content=content,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
        model="gpt-4o",
        tool_calls=tcs,
        finish_reason="tool_calls" if tcs else "stop",
    )


class MockTool(BaseTool):
    name = "mock_tool"
    description = "A mock tool for testing"
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    async def _execute(self, query: str) -> str:
        return f"Mock result for: {query}"


# ─── AgentState Tests ─────────────────────────────────────────────────────────

class TestAgentState:
    def test_create_initial_state_basic(self):
        state = create_initial_state(
            user_message="Hello",
            session_id="s-1",
            user_id="u-1",
        )
        assert state["agent_type"] == "react"
        assert state["step_count"] == 0
        assert state["max_steps"] == 10
        assert state["final_answer"] is None
        # Last message should be the user message
        assert isinstance(state["messages"][-1], HumanMessage)
        assert state["messages"][-1].content == "Hello"

    def test_create_initial_state_with_history(self):
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        state = create_initial_state(
            user_message="Follow up",
            session_id="s-1",
            user_id="u-1",
            conversation_history=history,
        )
        # Should have 2 history + 1 current = 3 messages
        assert len(state["messages"]) == 3
        assert isinstance(state["messages"][0], HumanMessage)
        assert isinstance(state["messages"][1], AIMessage)
        assert isinstance(state["messages"][2], HumanMessage)
        assert state["messages"][2].content == "Follow up"

    def test_create_initial_state_custom_max_steps(self):
        state = create_initial_state(
            user_message="Test",
            session_id="s-1",
            user_id="u-1",
            max_steps=3,
        )
        assert state["max_steps"] == 3


# ─── ReActAgent Routing Tests ─────────────────────────────────────────────────

class TestReActAgentRouting:
    def setup_method(self):
        """Create a ReActAgent with mocked dependencies."""
        from backend.agents.react_agent import ReActAgent

        self.mock_llm = MagicMock()
        self.mock_llm.complete = AsyncMock()
        self.registry = ToolRegistry()
        self.registry.register(MockTool())
        self.agent = ReActAgent(
            llm=self.mock_llm,
            tool_registry=self.registry,
            max_steps=5,
        )

    def test_routes_to_tools_when_tool_calls_pending(self):
        state = make_state(
            pending_tool_calls=[{"id": "c1", "name": "mock_tool", "args": {"query": "test"}}]
        )
        result = self.agent._should_use_tools(state)
        assert result == "tools"

    def test_routes_to_end_when_final_answer_set(self):
        state = make_state(final_answer="Here is my answer.")
        result = self.agent._should_use_tools(state)
        assert result == "end"

    def test_routes_to_tools_when_ai_message_has_tool_calls(self):
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "mock_tool", "args": {}, "type": "tool_call"}],
        )
        state = make_state(messages=[HumanMessage(content="hi"), ai_msg])
        result = self.agent._should_use_tools(state)
        assert result == "tools"

    def test_routes_to_end_when_no_tool_calls(self):
        state = make_state(
            messages=[HumanMessage(content="hi"), AIMessage(content="Final answer")],
            pending_tool_calls=[],
            final_answer=None,
        )
        result = self.agent._should_use_tools(state)
        assert result == "end"

    @pytest.mark.asyncio
    async def test_reasoning_node_final_answer(self):
        """LLM returns content with no tool calls → sets final_answer."""
        self.mock_llm.complete = AsyncMock(
            return_value=make_llm_response(content="The answer is 4.")
        )
        state = make_state()
        result = await self.agent._reasoning_node(state)

        assert result["final_answer"] == "The answer is 4."
        assert result["step_count"] == 1
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)

    @pytest.mark.asyncio
    async def test_reasoning_node_tool_call(self):
        """LLM returns tool_calls → sets pending_tool_calls."""
        from backend.llm.base import ToolCall

        mock_response = make_llm_response(
            content="",
            tool_calls=[ToolCall(id="call-1", name="mock_tool", arguments={"query": "test"})]
        )
        self.mock_llm.complete = AsyncMock(return_value=mock_response)

        state = make_state()
        result = await self.agent._reasoning_node(state)

        # When tool calls requested, final_answer is absent from partial update (LangGraph
        # merges partial dicts — omitting a key means keep existing value, not set to None)
        assert result.get("final_answer") is None
        assert len(result["pending_tool_calls"]) == 1
        assert result["pending_tool_calls"][0]["name"] == "mock_tool"

    @pytest.mark.asyncio
    async def test_reasoning_node_stops_at_max_steps(self):
        """Agent stops gracefully when max_steps reached."""
        state = make_state(step_count=5, max_steps=5)
        result = await self.agent._reasoning_node(state)

        assert result["final_answer"] is not None
        assert result["step_count"] == 6

    @pytest.mark.asyncio
    async def test_tool_node_executes_tool(self):
        """Tool node runs tool and appends ToolMessage to messages."""
        state = make_state(
            pending_tool_calls=[
                {"id": "call-1", "name": "mock_tool", "args": {"query": "hello"}}
            ]
        )
        result = await self.agent._tool_node(state)

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], ToolMessage)
        assert "Mock result" in result["messages"][0].content
        assert result["pending_tool_calls"] == []

    @pytest.mark.asyncio
    async def test_tool_node_handles_unknown_tool(self):
        """Unknown tool name produces an error ToolMessage, doesn't raise."""
        state = make_state(
            pending_tool_calls=[
                {"id": "call-1", "name": "nonexistent_tool", "args": {}}
            ]
        )
        result = await self.agent._tool_node(state)

        assert len(result["messages"]) == 1
        assert result["messages"][0].content is not None
        # Should contain error info
        assert "not found" in result["messages"][0].content.lower() or "nonexistent" in result["messages"][0].content.lower()

    @pytest.mark.asyncio
    async def test_tool_node_parallel_execution(self):
        """Multiple tool calls execute concurrently."""
        state = make_state(
            pending_tool_calls=[
                {"id": "call-1", "name": "mock_tool", "args": {"query": "first"}},
                {"id": "call-2", "name": "mock_tool", "args": {"query": "second"}},
            ]
        )
        result = await self.agent._tool_node(state)

        assert len(result["messages"]) == 2
        assert len(result["tool_results"]) == 2


# ─── AgentFactory Tests ───────────────────────────────────────────────────────

class TestAgentFactory:
    def setup_method(self):
        from backend.agents.orchestrator import AgentFactory

        mock_llm = MagicMock()
        registry = ToolRegistry()
        self.factory = AgentFactory(llm=mock_llm, tool_registry=registry)

    def test_creates_react_agent(self):
        from backend.agents.react_agent import ReActAgent
        agent = self.factory.get_agent("react")
        assert isinstance(agent, ReActAgent)

    def test_caches_agent_instances(self):
        agent1 = self.factory.get_agent("react")
        agent2 = self.factory.get_agent("react")
        assert agent1 is agent2  # Same instance

    def test_different_max_steps_different_instance(self):
        agent1 = self.factory.get_agent("react", max_steps=5)
        agent2 = self.factory.get_agent("react", max_steps=10)
        assert agent1 is not agent2

    def test_unknown_agent_type_raises(self):
        with pytest.raises(ValueError, match="Unknown agent type"):
            self.factory.get_agent("nonexistent_agent")

    def test_clear_cache(self):
        agent1 = self.factory.get_agent("react")
        self.factory.clear_cache()
        agent2 = self.factory.get_agent("react")
        assert agent1 is not agent2  # Different instance after clear


# ─── Integration: Full Agent Run ──────────────────────────────────────────────

class TestReActAgentFullRun:
    @pytest.mark.asyncio
    async def test_single_turn_no_tools(self):
        """Agent answers directly without any tool calls."""
        from backend.agents.react_agent import ReActAgent

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(
            return_value=make_llm_response(content="Paris is the capital of France.")
        )
        registry = ToolRegistry()
        agent = ReActAgent(llm=mock_llm, tool_registry=registry, max_steps=5)

        state = await agent.run(
            user_message="What is the capital of France?",
            session_id="s-1",
            user_id="u-1",
        )

        assert state["final_answer"] == "Paris is the capital of France."
        assert state["step_count"] == 1
        mock_llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_use_then_answer(self):
        """Agent calls a tool, gets result, then produces final answer."""
        from backend.agents.react_agent import ReActAgent
        from backend.llm.base import ToolCall

        call_count = 0

        async def mock_complete(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: request tool use
                return make_llm_response(
                    tool_calls=[ToolCall(id="c1", name="mock_tool", arguments={"query": "Paris population"})]
                )
            else:
                # Second call: produce final answer after seeing tool result
                return make_llm_response(content="Based on the search, Paris has ~2.1M people.")

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(side_effect=mock_complete)

        registry = ToolRegistry()
        registry.register(MockTool())
        agent = ReActAgent(llm=mock_llm, tool_registry=registry, max_steps=5)

        state = await agent.run(
            user_message="What is the population of Paris?",
            session_id="s-1",
            user_id="u-1",
        )

        assert state["final_answer"] is not None
        assert "Paris" in state["final_answer"]
        assert state["step_count"] == 2
        assert call_count == 2  # LLM called twice: once for tool, once for answer
        assert len(state["tool_results"]) == 1