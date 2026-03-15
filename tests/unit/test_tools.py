"""
Tool Framework Unit Tests
=========================
Tests for BaseTool, ToolRegistry, and all individual tools.
No external API calls — all tools are tested with mocked dependencies.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

from backend.tools.base import BaseTool, ToolResult
from backend.tools.registry import ToolRegistry, get_tools_for_agent, TOOL_PERMISSIONS


# ─── Helpers ──────────────────────────────────────────────────────────────────

class EchoTool(BaseTool):
    """Minimal test tool that echoes its input."""
    name = "echo"
    description = "Echoes the input text"
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
        },
        "required": ["text"],
    }

    async def _execute(self, text: str) -> str:
        return f"Echo: {text}"


class FailingTool(BaseTool):
    """Test tool that always raises an exception."""
    name = "failing_tool"
    description = "Always fails"
    parameters = {
        "type": "object",
        "properties": {
            "input": {"type": "string"},
        },
        "required": ["input"],
    }

    async def _execute(self, input: str) -> str:
        raise RuntimeError("Intentional failure for testing")


# ─── BaseTool Tests ───────────────────────────────────────────────────────────

class TestBaseTool:
    @pytest.mark.asyncio
    async def test_successful_execution(self):
        tool = EchoTool()
        result = await tool.run({"text": "hello world"})

        assert result.success is True
        assert result.output == "Echo: hello world"
        assert result.tool_name == "echo"
        assert result.latency_ms >= 0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_failed_execution_returns_toolresult(self):
        """Failures are wrapped in ToolResult, never raised."""
        tool = FailingTool()
        result = await tool.run({"input": "trigger failure"})

        assert result.success is False
        assert result.error == "Intentional failure for testing"
        assert result.output == ""
        assert result.tool_name == "failing_tool"

    @pytest.mark.asyncio
    async def test_missing_required_param_returns_error(self):
        tool = EchoTool()
        result = await tool.run({})  # Missing required "text"

        assert result.success is False
        assert "text" in result.error

    @pytest.mark.asyncio
    async def test_output_truncated_when_too_long(self):
        class LongOutputTool(BaseTool):
            name = "long_output"
            description = "Returns very long output"
            parameters = {"type": "object", "properties": {}, "required": []}
            max_output_length = 100  # Small limit for test

            async def _execute(self) -> str:
                return "x" * 500

        tool = LongOutputTool()
        result = await tool.run({})

        assert result.success is True
        assert len(result.output) <= 120  # 100 chars + truncation message
        assert "[Output truncated]" in result.output

    def test_to_llm_definition_format(self):
        tool = EchoTool()
        definition = tool.to_llm_definition()

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "echo"
        assert definition["function"]["description"] == tool.description
        assert "parameters" in definition["function"]

    def test_tool_result_to_llm_string_success(self):
        result = ToolResult(tool_name="echo", success=True, output="Hello!")
        assert result.to_llm_string() == "Hello!"

    def test_tool_result_to_llm_string_failure(self):
        result = ToolResult(
            tool_name="echo", success=False, output="", error="Something went wrong"
        )
        assert "echo" in result.to_llm_string()
        assert "Something went wrong" in result.to_llm_string()

    def test_missing_name_raises_on_class_definition(self):
        """Subclass without 'name' attribute should raise TypeError."""
        with pytest.raises(TypeError, match="must define 'name'"):
            class BadTool(BaseTool):
                description = "No name defined"
                parameters = {}

                async def _execute(self):
                    pass


# ─── ToolRegistry Tests ───────────────────────────────────────────────────────

class TestToolRegistry:
    def setup_method(self):
        """Fresh registry for each test."""
        self.registry = ToolRegistry()

    def test_register_and_get(self):
        tool = EchoTool()
        self.registry.register(tool)

        retrieved = self.registry.get("echo")
        assert retrieved is tool

    def test_get_nonexistent_returns_none(self):
        assert self.registry.get("nonexistent") is None

    def test_get_tools_returns_list(self):
        self.registry.register(EchoTool())
        self.registry.register(FailingTool())

        tools = self.registry.get_tools(["echo", "failing_tool"])
        assert len(tools) == 2

    def test_get_tools_skips_unknown(self):
        self.registry.register(EchoTool())
        tools = self.registry.get_tools(["echo", "nonexistent"])
        assert len(tools) == 1
        assert tools[0].name == "echo"

    def test_contains(self):
        self.registry.register(EchoTool())
        assert "echo" in self.registry
        assert "nonexistent" not in self.registry

    def test_len(self):
        assert len(self.registry) == 0
        self.registry.register(EchoTool())
        assert len(self.registry) == 1

    def test_to_llm_definitions(self):
        self.registry.register(EchoTool())
        defs = self.registry.to_llm_definitions(["echo"])
        assert len(defs) == 1
        assert defs[0]["function"]["name"] == "echo"

    def test_to_llm_definitions_all(self):
        self.registry.register(EchoTool())
        self.registry.register(FailingTool())
        defs = self.registry.to_llm_definitions()  # None = all
        assert len(defs) == 2

    @pytest.mark.asyncio
    async def test_execute_success(self):
        self.registry.register(EchoTool())
        result = await self.registry.execute("echo", {"text": "test"})
        assert result.success is True
        assert "test" in result.output

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        result = await self.registry.execute("nonexistent", {})
        assert result.success is False
        assert "not found" in result.error


# ─── Permission Tests ─────────────────────────────────────────────────────────

class TestToolPermissions:
    def test_all_agent_types_have_permissions(self):
        for agent_type in ["react", "researcher", "analyst", "writer", "admin"]:
            tools = get_tools_for_agent(agent_type)
            assert isinstance(tools, list)
            assert len(tools) > 0

    def test_unknown_agent_type_returns_default(self):
        tools = get_tools_for_agent("unknown_agent")
        # Should return react defaults, not error
        assert isinstance(tools, list)

    def test_admin_has_all_tools(self):
        admin_tools = set(get_tools_for_agent("admin"))
        react_tools = set(get_tools_for_agent("react"))
        # Admin should have at least everything react has
        assert react_tools.issubset(admin_tools)

    def test_writer_has_minimal_tools(self):
        # Writer only needs to read files, not search web
        writer_tools = get_tools_for_agent("writer")
        assert "web_search" not in writer_tools


# ─── Calculator Tool Tests ────────────────────────────────────────────────────

class TestCalculatorTool:
    def setup_method(self):
        from backend.tools.calculator import CalculatorTool
        self.tool = CalculatorTool()

    @pytest.mark.asyncio
    async def test_basic_arithmetic(self):
        result = await self.tool.run({"expression": "2 + 2"})
        assert result.success is True
        assert "4" in result.output

    @pytest.mark.asyncio
    async def test_multiplication(self):
        result = await self.tool.run({"expression": "6 * 7"})
        assert result.success is True
        assert "42" in result.output

    @pytest.mark.asyncio
    async def test_power_caret(self):
        """Caret ^ should be treated as ** (exponentiation)."""
        result = await self.tool.run({"expression": "2^10"})
        assert result.success is True
        assert "1024" in result.output

    @pytest.mark.asyncio
    async def test_sqrt(self):
        result = await self.tool.run({"expression": "sqrt(144)"})
        assert result.success is True
        assert "12" in result.output

    @pytest.mark.asyncio
    async def test_float_result(self):
        result = await self.tool.run({"expression": "10 / 3"})
        assert result.success is True
        assert "3.333" in result.output

    @pytest.mark.asyncio
    async def test_invalid_expression(self):
        result = await self.tool.run({"expression": "not_a_number + ?"})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_missing_expression(self):
        result = await self.tool.run({})
        assert result.success is False
        assert "expression" in result.error


# ─── Database Query Tool Tests ────────────────────────────────────────────────

class TestDatabaseQueryTool:
    def setup_method(self):
        from backend.tools.database_query import DatabaseQueryTool
        self.tool = DatabaseQueryTool()

    def test_blocks_insert(self):
        error = self.tool._validate_query("INSERT INTO users VALUES (1)")
        assert error is not None
        assert "SELECT" in error

    def test_blocks_drop(self):
        error = self.tool._validate_query("DROP TABLE users")
        assert error is not None

    def test_blocks_disallowed_table(self):
        error = self.tool._validate_query("SELECT * FROM pg_tables")
        assert error is not None

    def test_allows_valid_select(self):
        error = self.tool._validate_query("SELECT id, title FROM documents LIMIT 10")
        assert error is None

    def test_blocks_comment_injection(self):
        error = self.tool._validate_query(
            "SELECT * FROM documents; -- DROP TABLE documents"
        )
        assert error is not None

    def test_allows_all_permitted_tables(self):
        from backend.tools.database_query import ALLOWED_TABLES
        for table in ALLOWED_TABLES:
            error = self.tool._validate_query(f"SELECT * FROM {table} LIMIT 1")
            assert error is None, f"Table '{table}' should be allowed"


# ─── File Reader Tool Tests ───────────────────────────────────────────────────

class TestFileReaderTool:
    def setup_method(self):
        from backend.tools.file_reader import FileReaderTool
        self.tool = FileReaderTool()

    @pytest.mark.asyncio
    async def test_reads_file_successfully(self, tmp_path):
        """Test reading a file from the uploads directory."""
        from unittest.mock import patch

        # Override upload dir to tmp_path for test
        self.tool._upload_dir = tmp_path.resolve()

        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello from test file!")

        result = await self.tool.run({"filename": "test.txt"})
        assert result.success is True
        assert "Hello from test file!" in result.output

    @pytest.mark.asyncio
    async def test_blocks_path_traversal(self, tmp_path):
        self.tool._upload_dir = tmp_path.resolve()

        result = await self.tool.run({"filename": "../../etc/passwd"})
        assert result.success is False
        assert "Access denied" in result.error or "outside" in result.error.lower()

    @pytest.mark.asyncio
    async def test_file_not_found(self, tmp_path):
        self.tool._upload_dir = tmp_path.resolve()

        result = await self.tool.run({"filename": "nonexistent.txt"})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_rejects_unsupported_extension(self, tmp_path):
        self.tool._upload_dir = tmp_path.resolve()
        exe_file = tmp_path / "malicious.exe"
        exe_file.write_bytes(b"MZ binary content")

        result = await self.tool.run({"filename": "malicious.exe"})
        assert result.success is False