"""
Tests for Step 12 — MCP Integration.

All external HTTP calls mocked via unittest.mock.
No real API keys or network calls.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.mcp.client import (
    BaseMCPClient,
    MCPCapabilities,
    MCPServerConfig,
    MCPTool,
    MCPTransport,
)
from backend.mcp.connectors.google_drive import (
    GoogleDriveMCPClient,
    build_google_drive_config,
)
from backend.mcp.connectors.notion import NotionMCPClient, build_notion_config
from backend.mcp.connectors.slack import SlackMCPClient, build_slack_config
from backend.mcp.registry import MCPRegistry
from backend.tools.base import ToolResult


# ---------------------------------------------------------------------------
# Helpers — concrete stub of BaseMCPClient for testing
# ---------------------------------------------------------------------------


class _StubMCPClient(BaseMCPClient):
    """Minimal concrete client for testing BaseMCPClient behaviour."""

    def __init__(self, config: MCPServerConfig, should_connect: bool = True) -> None:
        super().__init__(config)
        self._should_connect = should_connect

    async def _connect_impl(self) -> bool:
        return self._should_connect

    async def _fetch_capabilities(self) -> MCPCapabilities:
        return MCPCapabilities(
            tools=[
                MCPTool(
                    name="echo",
                    description="Echo input",
                    input_schema={"type": "object"},
                    server_name=self.server_name,
                )
            ],
            server_name=self.server_name,
            available=True,
        )

    async def _call_tool_impl(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        if tool_name == "echo":
            return ToolResult(tool_name=tool_name, success=True, output=str(arguments))
        return ToolResult(tool_name=tool_name, success=False, output="", error="unknown tool")


def _make_config(name: str = "test", enabled: bool = True) -> MCPServerConfig:
    return MCPServerConfig(
        name=name,
        transport=MCPTransport.HTTP,
        url="http://fake-mcp-server",
        enabled=enabled,
    )


# ---------------------------------------------------------------------------
# BaseMCPClient tests
# ---------------------------------------------------------------------------


class TestBaseMCPClient:
    @pytest.mark.asyncio
    async def test_connect_success(self):
        client = _StubMCPClient(_make_config())
        result = await client.connect()
        assert result is True
        assert client.is_connected is True

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        client = _StubMCPClient(_make_config(), should_connect=False)
        result = await client.connect()
        assert result is False
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_disabled_config(self):
        client = _StubMCPClient(_make_config(enabled=False))
        result = await client.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_connect_idempotent(self):
        client = _StubMCPClient(_make_config())
        await client.connect()
        await client.connect()  # second call is a no-op
        assert client.is_connected is True

    @pytest.mark.asyncio
    async def test_disconnect(self):
        client = _StubMCPClient(_make_config())
        await client.connect()
        await client.disconnect()
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_list_capabilities_when_connected(self):
        client = _StubMCPClient(_make_config())
        await client.connect()
        caps = await client.list_capabilities()
        assert caps.available is True
        assert len(caps.tools) == 1
        assert caps.tools[0].name == "echo"

    @pytest.mark.asyncio
    async def test_list_capabilities_when_not_connected(self):
        client = _StubMCPClient(_make_config())
        caps = await client.list_capabilities()
        assert caps.available is False

    @pytest.mark.asyncio
    async def test_call_tool_when_connected(self):
        client = _StubMCPClient(_make_config())
        await client.connect()
        result = await client.call_tool("echo", {"input": "hello"})
        assert result.success is True
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_call_tool_when_not_connected(self):
        client = _StubMCPClient(_make_config())
        result = await client.call_tool("echo", {})
        assert result.success is False
        assert "not connected" in result.error.lower()

    @pytest.mark.asyncio
    async def test_call_tool_exception_handled(self):
        client = _StubMCPClient(_make_config())
        await client.connect()

        async def _boom(tool_name, arguments):
            raise ValueError("explosion")

        client._call_tool_impl = _boom
        result = await client.call_tool("echo", {})
        assert result.success is False
        assert "explosion" in result.error

    @pytest.mark.asyncio
    async def test_capabilities_cached(self):
        client = _StubMCPClient(_make_config())
        await client.connect()
        caps1 = await client.list_capabilities()
        caps2 = await client.list_capabilities()
        assert caps1 is caps2  # same object returned from cache


# ---------------------------------------------------------------------------
# MCPRegistry tests
# ---------------------------------------------------------------------------


class TestMCPRegistry:
    def _make_registry(self) -> MCPRegistry:
        return MCPRegistry()

    def test_register(self):
        registry = self._make_registry()
        registry.register(_make_config("srv1"), _StubMCPClient)
        assert "srv1" in registry.registered_servers

    def test_register_overwrite(self):
        registry = self._make_registry()
        registry.register(_make_config("srv1"), _StubMCPClient)
        registry.register(_make_config("srv1"), _StubMCPClient)
        assert registry.registered_servers.count("srv1") == 1

    def test_unregister(self):
        registry = self._make_registry()
        registry.register(_make_config("srv1"), _StubMCPClient)
        registry.unregister("srv1")
        assert "srv1" not in registry.registered_servers

    @pytest.mark.asyncio
    async def test_connect_all(self):
        registry = self._make_registry()
        registry.register(_make_config("srv1"), _StubMCPClient)
        registry.register(_make_config("srv2"), _StubMCPClient)
        results = await registry.connect_all()
        assert results["srv1"] is True
        assert results["srv2"] is True

    @pytest.mark.asyncio
    async def test_connect_all_with_failure(self):
        registry = self._make_registry()
        registry.register(_make_config("good"), _StubMCPClient)
        bad_config = MCPServerConfig(
            name="bad", transport=MCPTransport.HTTP, url="http://bad", enabled=True
        )

        class _FailClient(_StubMCPClient):
            async def _connect_impl(self) -> bool:
                return False

        registry.register(bad_config, _FailClient)
        results = await registry.connect_all()
        assert results["good"] is True
        assert results["bad"] is False

    @pytest.mark.asyncio
    async def test_disconnect_all(self):
        registry = self._make_registry()
        registry.register(_make_config("srv1"), _StubMCPClient)
        await registry.connect_all()
        await registry.disconnect_all()
        assert registry.connected_servers == []

    @pytest.mark.asyncio
    async def test_call_tool_dispatches_to_correct_server(self):
        registry = self._make_registry()
        registry.register(_make_config("test"), _StubMCPClient)
        await registry.connect_all()
        result = await registry.call_tool("test__echo", {"val": 42})
        assert result.success is True

    @pytest.mark.asyncio
    async def test_call_tool_unknown_server(self):
        registry = self._make_registry()
        result = await registry.call_tool("nonexistent__tool", {})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_call_tool_invalid_format(self):
        registry = self._make_registry()
        result = await registry.call_tool("notqualified", {})
        assert result.success is False
        assert "Invalid tool name" in result.error

    @pytest.mark.asyncio
    async def test_list_all_tools(self):
        registry = self._make_registry()
        registry.register(_make_config("srv1"), _StubMCPClient)
        registry.register(_make_config("srv2"), _StubMCPClient)
        await registry.connect_all()
        tools = await registry.list_all_tools()
        assert len(tools) == 2

    @pytest.mark.asyncio
    async def test_lazy_connect_on_call(self):
        registry = self._make_registry()
        registry.register(_make_config("lazy"), _StubMCPClient)
        # Don't call connect_all() — just call the tool
        result = await registry.call_tool("lazy__echo", {"x": 1})
        assert result.success is True
        assert registry.is_server_connected("lazy")

    @pytest.mark.asyncio
    async def test_capabilities_cache_invalidation(self):
        registry = self._make_registry()
        registry.register(_make_config("srv1"), _StubMCPClient)
        await registry.connect_all()
        await registry.list_all_tools()  # populates cache
        registry.invalidate_cache("srv1")
        assert "srv1" not in registry._capabilities_cache

    def test_parse_tool_name_valid(self):
        server, tool = MCPRegistry._parse_tool_name("google_drive__list_files")
        assert server == "google_drive"
        assert tool == "list_files"

    def test_parse_tool_name_invalid(self):
        server, tool = MCPRegistry._parse_tool_name("no_double_underscore")
        assert server is None
        assert tool is None


# ---------------------------------------------------------------------------
# Config factory tests
# ---------------------------------------------------------------------------


class TestConfigFactories:
    def test_google_drive_config_disabled_when_no_url(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_DRIVE_MCP_URL", raising=False)
        config = build_google_drive_config()
        assert config.enabled is False

    def test_google_drive_config_enabled_with_url(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_DRIVE_MCP_URL", "http://gdrive-mcp")
        config = build_google_drive_config()
        assert config.enabled is True
        assert config.url == "http://gdrive-mcp"

    def test_notion_config_disabled_when_no_url(self, monkeypatch):
        monkeypatch.delenv("NOTION_MCP_URL", raising=False)
        config = build_notion_config()
        assert config.enabled is False

    def test_slack_config_disabled_when_no_url(self, monkeypatch):
        monkeypatch.delenv("SLACK_MCP_URL", raising=False)
        config = build_slack_config()
        assert config.enabled is False

    def test_google_drive_token_in_headers(self):
        config = build_google_drive_config(url="http://x", token="mytoken")
        assert "Authorization" in config.headers
        assert "mytoken" in config.headers["Authorization"]


# ---------------------------------------------------------------------------
# Connector unit tests (HTTP calls mocked)
# ---------------------------------------------------------------------------


class TestGoogleDriveMCPClient:
    def _make_client(self) -> GoogleDriveMCPClient:
        config = build_google_drive_config(url="http://fake-gdrive", token="tok")
        return GoogleDriveMCPClient(config)

    @pytest.mark.asyncio
    async def test_connect_no_url(self):
        config = build_google_drive_config()  # no URL
        client = GoogleDriveMCPClient(config)
        result = await client.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_connect_success(self):
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"result": {"protocolVersion": "2024-11-05"}}

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            result = await client.connect()
        assert result is True

    @pytest.mark.asyncio
    async def test_fetch_capabilities(self):
        client = self._make_client()
        client._connected = True
        caps = await client._fetch_capabilities()
        assert caps.available is True
        assert any(t.name == "list_files" for t in caps.tools)

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        client = self._make_client()
        client._connected = True

        mock_http = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "content": [{"type": "text", "text": "file1.txt\nfile2.txt"}],
                "isError": False,
            }
        }
        mock_http.post = AsyncMock(return_value=mock_response)
        client._http_client = mock_http

        result = await client.call_tool("list_files", {"folder_id": "root"})
        assert result.success is True
        assert "file1.txt" in result.output


class TestNotionMCPClient:
    def _make_client(self) -> NotionMCPClient:
        config = build_notion_config(url="http://fake-notion", api_key="secret")
        return NotionMCPClient(config)

    @pytest.mark.asyncio
    async def test_connect_no_url(self):
        config = build_notion_config()
        client = NotionMCPClient(config)
        result = await client.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_fetch_capabilities(self):
        client = self._make_client()
        client._connected = True
        caps = await client._fetch_capabilities()
        assert caps.available is True
        tool_names = [t.name for t in caps.tools]
        assert "search_pages" in tool_names
        assert "create_page" in tool_names

    @pytest.mark.asyncio
    async def test_call_tool_error_response(self):
        client = self._make_client()
        client._connected = True

        mock_http = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "content": [{"type": "text", "text": "Page not found"}],
                "isError": True,
            }
        }
        mock_http.post = AsyncMock(return_value=mock_response)
        client._http_client = mock_http

        result = await client.call_tool("read_page", {"page_id": "nonexistent"})
        assert result.success is False
        assert "not found" in result.error.lower()


class TestSlackMCPClient:
    def _make_client(self) -> SlackMCPClient:
        config = build_slack_config(url="http://fake-slack", bot_token="xoxb-fake")
        return SlackMCPClient(config)

    @pytest.mark.asyncio
    async def test_connect_no_url(self):
        config = build_slack_config()
        client = SlackMCPClient(config)
        result = await client.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_fetch_capabilities(self):
        client = self._make_client()
        client._connected = True
        caps = await client._fetch_capabilities()
        assert caps.available is True
        tool_names = [t.name for t in caps.tools]
        assert "send_message" in tool_names
        assert "search_messages" in tool_names

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        client = self._make_client()
        client._connected = True

        mock_http = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "content": [{"type": "text", "text": "Message sent"}],
                "isError": False,
            }
        }
        mock_http.post = AsyncMock(return_value=mock_response)
        client._http_client = mock_http

        result = await client.call_tool("send_message", {"channel_id": "C123", "text": "hello"})
        assert result.success is True


# ---------------------------------------------------------------------------
# setup_mcp_registry tests
# ---------------------------------------------------------------------------


class TestSetupMCPRegistry:
    @pytest.mark.asyncio
    async def test_setup_no_env_vars(self, monkeypatch):
        """With no env vars, registry should have no registered servers."""
        for var in ("GOOGLE_DRIVE_MCP_URL", "NOTION_MCP_URL", "SLACK_MCP_URL"):
            monkeypatch.delenv(var, raising=False)

        import backend.mcp.registry as reg_module
        reg_module._registry = None  # reset singleton

        from backend.mcp import setup_mcp_registry
        registry = await setup_mcp_registry()
        assert registry.registered_servers == []

        reg_module._registry = None  # cleanup