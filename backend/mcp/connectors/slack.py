"""
Slack MCP Connector.

Design rationale:
- Wraps a Slack MCP server (e.g., https://github.com/modelcontextprotocol/servers/tree/main/src/slack)
- Exposes tools: list_channels, read_channel, send_message, search_messages,
  get_thread, list_users
- Bot token is forwarded to the MCP server — never stored in this process
- Enabled only when SLACK_MCP_URL is configured

Environment variables:
    SLACK_MCP_URL     — URL of running Slack MCP server (required to enable)
    SLACK_BOT_TOKEN   — Slack bot OAuth token forwarded as bearer auth
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from backend.mcp.client import (
    BaseMCPClient,
    MCPCapabilities,
    MCPServerConfig,
    MCPTool,
    MCPTransport,
)
from backend.tools.base import ToolResult

logger = logging.getLogger(__name__)

_SLACK_TOOLS: List[MCPTool] = [
    MCPTool(
        name="list_channels",
        description="List public and private Slack channels the bot has access to.",
        input_schema={
            "type": "object",
            "properties": {
                "types": {
                    "type": "string",
                    "description": "Comma-separated channel types: public_channel,private_channel,mpim,im",
                },
                "limit": {"type": "integer", "description": "Max channels to return (default 100)"},
            },
        },
        server_name="slack",
    ),
    MCPTool(
        name="read_channel",
        description="Read recent messages from a Slack channel.",
        input_schema={
            "type": "object",
            "properties": {
                "channel_id": {"type": "string", "description": "Slack channel ID (e.g. C01ABC123)"},
                "limit": {"type": "integer", "description": "Number of messages (default 50)"},
                "oldest": {"type": "string", "description": "Start of time range (Unix timestamp)"},
            },
            "required": ["channel_id"],
        },
        server_name="slack",
    ),
    MCPTool(
        name="send_message",
        description="Send a message to a Slack channel or user.",
        input_schema={
            "type": "object",
            "properties": {
                "channel_id": {"type": "string", "description": "Channel or user ID"},
                "text": {"type": "string", "description": "Message text (supports Slack mrkdwn)"},
                "thread_ts": {
                    "type": "string",
                    "description": "Reply in thread (provide parent message timestamp)",
                },
            },
            "required": ["channel_id", "text"],
        },
        server_name="slack",
    ),
    MCPTool(
        name="search_messages",
        description="Full-text search across all Slack messages.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "count": {"type": "integer", "description": "Max results (default 20)"},
            },
            "required": ["query"],
        },
        server_name="slack",
    ),
    MCPTool(
        name="get_thread",
        description="Get all replies in a Slack thread.",
        input_schema={
            "type": "object",
            "properties": {
                "channel_id": {"type": "string", "description": "Channel ID"},
                "thread_ts": {"type": "string", "description": "Parent message timestamp"},
            },
            "required": ["channel_id", "thread_ts"],
        },
        server_name="slack",
    ),
    MCPTool(
        name="list_users",
        description="List workspace members.",
        input_schema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max users (default 200)"},
            },
        },
        server_name="slack",
    ),
]


def build_slack_config(
    url: Optional[str] = None,
    bot_token: Optional[str] = None,
) -> MCPServerConfig:
    """Factory to build Slack MCPServerConfig from env or explicit args."""
    resolved_url = url or os.getenv("SLACK_MCP_URL", "")
    resolved_token = bot_token or os.getenv("SLACK_BOT_TOKEN", "")

    headers: Dict[str, str] = {}
    if resolved_token:
        headers["Authorization"] = f"Bearer {resolved_token}"

    return MCPServerConfig(
        name="slack",
        transport=MCPTransport.HTTP,
        url=resolved_url,
        headers=headers,
        timeout=30.0,
        enabled=bool(resolved_url),
    )


class SlackMCPClient(BaseMCPClient):
    """
    MCP client for Slack.

    All requests are forwarded as JSON-RPC 2.0 to a running Slack MCP server.
    """

    def __init__(self, config: MCPServerConfig) -> None:
        super().__init__(config)
        self._http_client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # BaseMCPClient implementation
    # ------------------------------------------------------------------

    async def _connect_impl(self) -> bool:
        if not self._config.url:
            logger.info("SlackMCPClient: SLACK_MCP_URL not set — connector disabled")
            return False

        self._http_client = httpx.AsyncClient(
            base_url=self._config.url,
            headers=self._make_headers(),
            timeout=self._config.timeout,
        )

        try:
            await self._rpc("initialize", {"protocolVersion": "2024-11-05"})
            return True
        except Exception as exc:
            logger.warning("SlackMCPClient probe failed: %s", exc)
            await self._http_client.aclose()
            self._http_client = None
            return False

    async def _disconnect_impl(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def _fetch_capabilities(self) -> MCPCapabilities:
        return MCPCapabilities(
            tools=_SLACK_TOOLS,
            resources=[],
            prompts=[],
            server_name="slack",
            server_version="1.0.0",
            available=True,
        )

    async def _call_tool_impl(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> ToolResult:
        result = await self._rpc(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
        )
        content = result.get("content", [])
        text_parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        output = "\n".join(text_parts) or self._parse_tool_result(result)
        is_error = result.get("isError", False)
        return ToolResult(
            tool_name=tool_name,
            success=not is_error,
            output=output if not is_error else "",
            error=output if is_error else None,
        )

    # ------------------------------------------------------------------
    # JSON-RPC helper
    # ------------------------------------------------------------------

    async def _rpc(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self._http_client:
            raise RuntimeError("HTTP client not initialised")
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }
        response = await self._http_client.post("/", json=payload)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise RuntimeError(
                f"Slack MCP error {data['error'].get('code')}: {data['error'].get('message')}"
            )
        return data.get("result", {})