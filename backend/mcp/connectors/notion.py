"""
Notion MCP Connector.

Design rationale:
- Wraps a Notion MCP server (https://github.com/makenotion/notion-mcp-server)
  or any compatible implementation
- Exposes tools: search_pages, read_page, create_page, append_to_page,
  list_databases, query_database
- Token is the Notion integration token forwarded to the MCP server
- Graceful no-op when NOTION_MCP_URL is not configured

Environment variables:
    NOTION_MCP_URL    — URL of running Notion MCP server (required to enable)
    NOTION_API_KEY    — Notion integration token forwarded as bearer auth
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

_NOTION_TOOLS: List[MCPTool] = [
    MCPTool(
        name="search_pages",
        description="Search Notion workspace pages and databases by keyword.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keyword"},
                "filter_type": {
                    "type": "string",
                    "enum": ["page", "database"],
                    "description": "Optional: limit to pages or databases",
                },
            },
            "required": ["query"],
        },
        server_name="notion",
    ),
    MCPTool(
        name="read_page",
        description="Read the full content of a Notion page by its page ID.",
        input_schema={
            "type": "object",
            "properties": {
                "page_id": {"type": "string", "description": "Notion page ID (UUID)"},
            },
            "required": ["page_id"],
        },
        server_name="notion",
    ),
    MCPTool(
        name="create_page",
        description="Create a new Notion page in a parent page or database.",
        input_schema={
            "type": "object",
            "properties": {
                "parent_id": {"type": "string", "description": "Parent page or database ID"},
                "title": {"type": "string", "description": "Page title"},
                "content": {"type": "string", "description": "Markdown content for the page body"},
            },
            "required": ["parent_id", "title"],
        },
        server_name="notion",
    ),
    MCPTool(
        name="append_to_page",
        description="Append text blocks to an existing Notion page.",
        input_schema={
            "type": "object",
            "properties": {
                "page_id": {"type": "string", "description": "Notion page ID"},
                "content": {"type": "string", "description": "Markdown content to append"},
            },
            "required": ["page_id", "content"],
        },
        server_name="notion",
    ),
    MCPTool(
        name="list_databases",
        description="List all databases accessible in the Notion workspace.",
        input_schema={"type": "object", "properties": {}},
        server_name="notion",
    ),
    MCPTool(
        name="query_database",
        description="Query a Notion database with optional filters.",
        input_schema={
            "type": "object",
            "properties": {
                "database_id": {"type": "string", "description": "Notion database ID"},
                "filter": {
                    "type": "object",
                    "description": "Notion API filter object (optional)",
                },
                "max_results": {"type": "integer", "description": "Max rows to return (default 20)"},
            },
            "required": ["database_id"],
        },
        server_name="notion",
    ),
]


def build_notion_config(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> MCPServerConfig:
    """Factory to build Notion MCPServerConfig from env or explicit args."""
    resolved_url = url or os.getenv("NOTION_MCP_URL", "")
    resolved_key = api_key or os.getenv("NOTION_API_KEY", "")

    headers: Dict[str, str] = {}
    if resolved_key:
        headers["Authorization"] = f"Bearer {resolved_key}"

    return MCPServerConfig(
        name="notion",
        transport=MCPTransport.HTTP,
        url=resolved_url,
        headers=headers,
        timeout=30.0,
        enabled=bool(resolved_url),
    )


class NotionMCPClient(BaseMCPClient):
    """
    MCP client for Notion.

    Sends JSON-RPC 2.0 requests to a Notion MCP server over HTTP.
    """

    def __init__(self, config: MCPServerConfig) -> None:
        super().__init__(config)
        self._http_client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # BaseMCPClient implementation
    # ------------------------------------------------------------------

    async def _connect_impl(self) -> bool:
        if not self._config.url:
            logger.info("NotionMCPClient: NOTION_MCP_URL not set — connector disabled")
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
            logger.warning("NotionMCPClient probe failed: %s", exc)
            await self._http_client.aclose()
            self._http_client = None
            return False

    async def _disconnect_impl(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def _fetch_capabilities(self) -> MCPCapabilities:
        return MCPCapabilities(
            tools=_NOTION_TOOLS,
            resources=[],
            prompts=[],
            server_name="notion",
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
                f"Notion MCP error {data['error'].get('code')}: {data['error'].get('message')}"
            )
        return data.get("result", {})