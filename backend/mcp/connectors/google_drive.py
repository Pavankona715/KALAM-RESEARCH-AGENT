"""
Google Drive MCP Connector.

Design rationale:
- Uses the official Google Drive MCP server via SSE or HTTP transport
- Falls back gracefully if credentials are missing (GOOGLE_DRIVE_MCP_URL not set)
- Exposes tools: list_files, read_file, search_files, create_file
- All OAuth token handling is delegated to the MCP server — this client
  only sends JSON-RPC calls via httpx, keeping secrets out of this process

Environment variables:
    GOOGLE_DRIVE_MCP_URL  — URL of running Google Drive MCP server (required)
    GOOGLE_DRIVE_TOKEN    — Bearer token / API key for the MCP server (optional)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import httpx

from backend.mcp.client import (
    BaseMCPClient,
    MCPCapabilities,
    MCPResource,
    MCPServerConfig,
    MCPTool,
    MCPTransport,
)
from backend.tools.base import ToolResult

logger = logging.getLogger(__name__)

# Pre-defined schema so we can expose tools even before a live connection
_GOOGLE_DRIVE_TOOLS = [
    MCPTool(
        name="list_files",
        description="List files in Google Drive. Optionally filter by folder or MIME type.",
        input_schema={
            "type": "object",
            "properties": {
                "folder_id": {
                    "type": "string",
                    "description": "Google Drive folder ID (default: root)",
                },
                "mime_type": {
                    "type": "string",
                    "description": "Filter by MIME type e.g. application/vnd.google-apps.document",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of files to return (default 20)",
                },
            },
        },
        server_name="google_drive",
    ),
    MCPTool(
        name="read_file",
        description="Read the content of a Google Drive file by its file ID.",
        input_schema={
            "type": "object",
            "properties": {
                "file_id": {"type": "string", "description": "Google Drive file ID"},
            },
            "required": ["file_id"],
        },
        server_name="google_drive",
    ),
    MCPTool(
        name="search_files",
        description="Full-text search across Google Drive files.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string"},
                "max_results": {"type": "integer", "description": "Max results (default 10)"},
            },
            "required": ["query"],
        },
        server_name="google_drive",
    ),
    MCPTool(
        name="create_file",
        description="Create a new text file in Google Drive.",
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "File name"},
                "content": {"type": "string", "description": "File text content"},
                "folder_id": {"type": "string", "description": "Parent folder ID (optional)"},
            },
            "required": ["name", "content"],
        },
        server_name="google_drive",
    ),
]


def build_google_drive_config(
    url: Optional[str] = None,
    token: Optional[str] = None,
) -> MCPServerConfig:
    """Factory to build GoogleDrive MCPServerConfig from env or explicit args."""
    resolved_url = url or os.getenv("GOOGLE_DRIVE_MCP_URL", "")
    resolved_token = token or os.getenv("GOOGLE_DRIVE_TOKEN", "")

    headers: Dict[str, str] = {}
    if resolved_token:
        headers["Authorization"] = f"Bearer {resolved_token}"

    return MCPServerConfig(
        name="google_drive",
        transport=MCPTransport.HTTP,
        url=resolved_url,
        headers=headers,
        timeout=30.0,
        enabled=bool(resolved_url),
    )


class GoogleDriveMCPClient(BaseMCPClient):
    """
    MCP client for Google Drive.

    Communicates with a running Google Drive MCP server via HTTP JSON-RPC.
    If the server URL is not configured, connect() returns False immediately
    and all tool calls return descriptive errors.
    """

    def __init__(self, config: MCPServerConfig) -> None:
        super().__init__(config)
        self._http_client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # BaseMCPClient implementation
    # ------------------------------------------------------------------

    async def _connect_impl(self) -> bool:
        if not self._config.url:
            logger.info(
                "GoogleDriveMCPClient: GOOGLE_DRIVE_MCP_URL not set — connector disabled"
            )
            return False

        self._http_client = httpx.AsyncClient(
            base_url=self._config.url,
            headers=self._make_headers(),
            timeout=self._config.timeout,
        )

        # Probe the server with an initialize call
        try:
            await self._rpc("initialize", {"protocolVersion": "2024-11-05"})
            return True
        except Exception as exc:
            logger.warning("GoogleDriveMCPClient probe failed: %s", exc)
            await self._http_client.aclose()
            self._http_client = None
            return False

    async def _disconnect_impl(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def _fetch_capabilities(self) -> MCPCapabilities:
        # Use pre-defined schemas; optionally merge with server-reported tools
        return MCPCapabilities(
            tools=_GOOGLE_DRIVE_TOOLS,
            resources=[],
            prompts=[],
            server_name="google_drive",
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

    async def _read_resource_impl(self, uri: str) -> str:
        result = await self._rpc("resources/read", {"uri": uri})
        contents = result.get("contents", [])
        parts = [
            c.get("text", "")
            for c in contents
            if isinstance(c, dict) and c.get("type") == "text"
        ]
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # JSON-RPC helper
    # ------------------------------------------------------------------

    async def _rpc(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC 2.0 request and return the result dict."""
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
                f"MCP RPC error {data['error'].get('code')}: {data['error'].get('message')}"
            )
        return data.get("result", {})