"""
Multi-Agent Workflow Tests
==========================
Tests for individual agents and the full workflow pipeline.
All LLM calls are mocked — no real API calls needed.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.agents.multi_agent_state import MultiAgentState, create_multi_agent_state
from backend.llm.base import LLMResponse, TokenUsage
from backend.tools.registry import ToolRegistry
from backend.tools.base import BaseTool, ToolResult


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
        model="gpt-4o",
        finish_reason="stop",
    )


def make_mock_llm(response_content: str = "Mock LLM response") -> MagicMock:
    from backend.llm.router import LLMRouter
    llm = MagicMock(spec=LLMRouter)
    llm.complete = AsyncMock(return_value=make_llm_response(response_content))
    return llm


def make_base_state(**overrides) -> MultiAgentState:
    state = create_multi_agent_state(
        user_request="What is the current state of AI?",
        session_id="sess-1",
        user_id="user-1",
        run_id="run-1",
    )
    state.update(overrides)
    return state


# ─── MultiAgentState Tests ────────────────────────────────────────────────────

class TestMultiAgentState:
    def test_create_initial_state(self):
        state = create_multi_agent_state(
            user_request="Research quantum computing",
            session_id="s-1",
            user_id="u-1",
        )
        assert state["user_request"] == "Research quantum computing"
        assert state["plan"] is None
        assert state["research_findings"] is None
        assert state["analysis"] is None
        assert state["final_report"] is None
        assert state["current_agent"] == "planner"
        assert state["completed_agents"] == []

    def test_initial_messages_contain_user_request(self):
        from langchain_core.messages import HumanMessage
        state = create_multi_agent_state(
            user_request="Test request",
            session_id="s-1",
            user_id="u-1",
        )
        assert len(state["messages"]) == 1
        assert isinstance(state["messages"][0], HumanMessage)
        assert state["messages"][0].content == "Test request"


# ─── PlannerAgent Tests ───────────────────────────────────────────────────────

class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_planner_produces_plan(self):
        from backend.agents.planner import PlannerAgent
        import json

        plan_json = json.dumps({
            "objective": "Research AI developments",
            "subtasks": [
                {"id": "t1", "description": "Search for AI news", "agent": "researcher"}
            ],
            "output_format": "report",
            "estimated_complexity": "medium",
        })

        llm = make_mock_llm(plan_json)
        planner = PlannerAgent(llm=llm)
        state = make_base_state()

        result = await planner.run(state)

        assert "plan" in result
        assert result["plan"]["objective"] == "Research AI developments"
        assert len(result["plan"]["subtasks"]) == 1
        assert "planner" in result["completed_agents"]
        assert result["current_agent"] == "researcher"

    @pytest.mark.asyncio
    async def test_planner_handles_invalid_json(self):
        """Planner uses fallback plan when LLM returns invalid JSON."""
        from backend.agents.planner import PlannerAgent

        llm = make_mock_llm("This is not JSON at all")
        planner = PlannerAgent(llm=llm)
        state = make_base_state()

        result = await planner.run(state)

        # Should have a valid fallback plan
        assert "plan" in result
        assert result["plan"]["objective"] == state["user_request"]
        assert len(result["plan"]["subtasks"]) > 0

    @pytest.mark.asyncio
    async def test_planner_handles_llm_failure(self):
        """Planner uses fallback plan on LLM error."""
        from backend.agents.planner import PlannerAgent
        from backend.llm.router import LLMRouter

        llm = MagicMock(spec=LLMRouter)
        llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))
        planner = PlannerAgent(llm=llm)
        state = make_base_state()

        result = await planner.run(state)

        # Should not raise — falls back to default plan
        assert "plan" in result
        assert result["plan"] is not None

    @pytest.mark.asyncio
    async def test_planner_sets_next_agent(self):
        from backend.agents.planner import PlannerAgent
        import json

        llm = make_mock_llm(json.dumps({
            "objective": "Test", "subtasks": [], "output_format": "answer"
        }))
        planner = PlannerAgent(llm=llm)
        result = await planner.run(make_base_state())

        assert result["current_agent"] == "researcher"


# ─── ResearchAgent Tests ──────────────────────────────────────────────────────

class TestResearchAgent:
    def _make_researcher(self, response_content: str = "Research findings here"):
        from backend.agents.researcher import ResearchAgent
        llm = make_mock_llm(response_content)
        registry = ToolRegistry()
        return ResearchAgent(llm=llm, tool_registry=registry)

    @pytest.mark.asyncio
    async def test_researcher_produces_findings(self):
        researcher = self._make_researcher("Found: AI is advancing rapidly.")
        state = make_base_state(plan={
            "objective": "Research AI",
            "subtasks": [{"description": "Search AI news", "agent": "researcher"}],
            "output_format": "report",
        })

        result = await researcher.run(state)

        assert "research_findings" in result
        assert result["research_findings"] is not None
        assert "researcher" in result["completed_agents"]
        assert result["current_agent"] == "analyst"

    @pytest.mark.asyncio
    async def test_researcher_works_without_plan(self):
        """Researcher handles missing plan gracefully."""
        researcher = self._make_researcher("General findings.")
        state = make_base_state(plan=None)

        result = await researcher.run(state)

        assert result["research_findings"] is not None

    @pytest.mark.asyncio
    async def test_researcher_collects_sources_from_tools(self):
        """When tools are available and return results, sources are collected."""
        from backend.agents.researcher import ResearchAgent

        # Mock LLM that requests a tool call first, then gives final answer
        from backend.llm.base import ToolCall
        tool_call_response = LLMResponse(
            content="",
            usage=TokenUsage(10, 5),
            model="gpt-4o",
            tool_calls=[ToolCall(id="c1", name="web_search", arguments={"query": "AI 2024"})],
            finish_reason="tool_calls",
        )
        final_response = make_llm_response("AI developments in 2024 include...")

        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=[tool_call_response, final_response])

        registry = ToolRegistry()

        class MockSearchTool(BaseTool):
            name = "web_search"
            description = "Search"
            parameters = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
            async def _execute(self, query: str) -> str:
                return "Search result: AI is progressing"

        registry.register(MockSearchTool())
        researcher = ResearchAgent(llm=llm, tool_registry=registry)

        state = make_base_state(plan={
            "objective": "Research AI",
            "subtasks": [{"description": "Search AI", "agent": "researcher"}],
            "output_format": "report",
        })

        result = await researcher.run(state)
        assert result["research_findings"] is not None


# ─── AnalystAgent Tests ───────────────────────────────────────────────────────

class TestAnalystAgent:
    @pytest.mark.asyncio
    async def test_analyst_produces_analysis(self):
        from backend.agents.analyst import AnalystAgent

        analysis_text = (
            "Analysis of the findings:\n"
            "1. AI is advancing rapidly\n"
            "2. New models are more capable\n"
            "3. Safety remains a concern\n"
        )
        llm = make_mock_llm(analysis_text)
        analyst = AnalystAgent(llm=llm)

        state = make_base_state(
            research_findings="LLMs have improved dramatically in 2024.",
            plan={"objective": "Research AI", "output_format": "report", "subtasks": []},
        )

        result = await analyst.run(state)

        assert "analysis" in result
        assert result["analysis"] == analysis_text
        assert "analyst" in result["completed_agents"]
        assert result["current_agent"] == "writer"

    @pytest.mark.asyncio
    async def test_analyst_extracts_key_insights(self):
        from backend.agents.analyst import AnalystAgent

        llm = make_mock_llm(
            "Overview here.\n"
            "1. First key insight about AI\n"
            "2. Second key insight about safety\n"
            "- Additional bullet point insight\n"
        )
        analyst = AnalystAgent(llm=llm)
        state = make_base_state(research_findings="Some findings")

        result = await analyst.run(state)

        assert len(result["key_insights"]) >= 2

    @pytest.mark.asyncio
    async def test_analyst_handles_missing_findings(self):
        from backend.agents.analyst import AnalystAgent

        llm = make_mock_llm("Analysis based on limited data.")
        analyst = AnalystAgent(llm=llm)
        state = make_base_state(research_findings=None)

        result = await analyst.run(state)
        assert result["analysis"] is not None


# ─── WriterAgent Tests ────────────────────────────────────────────────────────

class TestWriterAgent:
    @pytest.mark.asyncio
    async def test_writer_produces_final_report(self):
        from backend.agents.writer import WriterAgent

        llm = make_mock_llm("# Final Report\n\nThis is the final output.")
        writer = WriterAgent(llm=llm)

        state = make_base_state(
            research_findings="Key research here",
            analysis="Key analysis here",
            key_insights=["Insight 1", "Insight 2"],
            plan={"objective": "Research AI", "output_format": "report", "subtasks": []},
        )

        result = await writer.run(state)

        assert "final_report" in result
        assert result["final_report"] == "# Final Report\n\nThis is the final output."
        assert "writer" in result["completed_agents"]
        assert result["current_agent"] == "done"

    @pytest.mark.asyncio
    async def test_writer_uses_analysis_as_fallback(self):
        from backend.agents.writer import WriterAgent
        from backend.llm.router import LLMRouter

        llm = MagicMock(spec=LLMRouter)
        llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))
        writer = WriterAgent(llm=llm)

        state = make_base_state(
            analysis="Fallback analysis content",
            research_findings="Some findings",
        )

        result = await writer.run(state)
        assert result["final_report"] is not None


# ─── Full Workflow Tests ──────────────────────────────────────────────────────

class TestMultiAgentWorkflow:
    def _make_workflow(self):
        from backend.agents.multi_agent_workflow import MultiAgentWorkflow
        import json

        plan_json = json.dumps({
            "objective": "Research AI advancements",
            "subtasks": [
                {"id": "t1", "description": "Search AI news", "agent": "researcher"}
            ],
            "output_format": "report",
            "estimated_complexity": "low",
        })

        responses = [
            plan_json,                              # planner
            "Key findings: AI is advancing.",       # researcher
            "1. AI models are improving\n2. Safety matters.",  # analyst
            "# Final Report\n\nAI has advanced significantly in 2024.",  # writer
        ]

        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=[make_llm_response(r) for r in responses])

        registry = ToolRegistry()
        return MultiAgentWorkflow(llm=llm, tool_registry=registry), llm

    @pytest.mark.asyncio
    async def test_full_pipeline_runs_all_agents(self):
        workflow, _ = self._make_workflow()

        result = await workflow.run(
            user_request="What is the state of AI in 2024?",
            session_id="s-1",
            user_id="u-1",
        )

        assert set(result["completed_agents"]) == {"planner", "researcher", "analyst", "writer"}
        assert result["final_report"] is not None
        assert len(result["final_report"]) > 0

    @pytest.mark.asyncio
    async def test_full_pipeline_populates_all_fields(self):
        workflow, _ = self._make_workflow()

        result = await workflow.run(
            user_request="Research quantum computing",
            session_id="s-1",
            user_id="u-1",
        )

        assert result["plan"] is not None
        assert result["research_findings"] is not None
        assert result["analysis"] is not None
        assert result["final_report"] is not None

    @pytest.mark.asyncio
    async def test_llm_called_once_per_agent(self):
        workflow, mock_llm = self._make_workflow()

        await workflow.run(
            user_request="Test request",
            session_id="s-1",
            user_id="u-1",
        )

        # 4 agents × 1 LLM call each = 4 calls minimum
        assert mock_llm.complete.call_count >= 4

    @pytest.mark.asyncio
    async def test_workflow_with_retrieved_docs(self):
        workflow, _ = self._make_workflow()

        retrieved_docs = [
            {"content": "RAG document content", "source": "doc.pdf", "score": 0.9}
        ]

        result = await workflow.run(
            user_request="What does the document say?",
            session_id="s-1",
            user_id="u-1",
            retrieved_docs=retrieved_docs,
        )

        assert result["final_report"] is not None
        # Retrieved docs should be in metadata
        assert "retrieved_docs" in result.get("metadata", {})