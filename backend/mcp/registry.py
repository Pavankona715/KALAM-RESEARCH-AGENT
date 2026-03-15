"""
MCP Server Registry.

Design rationale:
- Central hub that owns all MCPClient instances
- Provides a unified interface so agents/routes never know which server a tool
  belongs to — they just call registry.call_tool("servername__toolname", args)
- Lazy connection: servers connect on first use, not at import time
- Thread-safe async initialization using asyncio.Lock per server
- Graceful degradation: if a server fails, others continue working
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Type

from backend.mcp.client import (
    BaseMCPClient,
    MCPCapabilities,
    MCPServerConfig,
    MCPTool,
    MCPTransport,
)
from backend.tools.base import ToolResult

logger = logging.getLogger(__name__)

# Global singleton registry (set up during FastAPI lifespan)
_registry: Optional["MCPRegistry"] = None


def get_mcp_registry() -> "MCPRegistry":
    """FastAPI dependency — returns the global registry (never raises)."""
    global _registry
    if _registry is None:
        _registry = MCPRegistry()
    return _registry


class MCPRegistry:
    """
    Manages a pool of MCP clients.

    Usage:
        registry = MCPRegistry()
        registry.register(config, GoogleDriveMCPClient)
        await registry.connect_all()
        result = await registry.call_tool("google_drive__list_files", {})
        await registry.disconnect_all()
    """

    def __init__(self) -> None:
        self._clients: Dict[str, BaseMCPClient] = {}
        self._connect_locks: Dict[str, asyncio.Lock] = {}
        self._capabilities_cache: Dict[str, MCPCapabilities] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        config: MCPServerConfig,
        client_class: Type[BaseMCPClient],
    ) -> None:
        """
        Register an MCP server.

        Args:
            config: Server configuration.
            client_class: Concrete BaseMCPClient subclass to instantiate.
        """
        if config.name in self._clients:
            logger.warning(
                "MCP server '%s' already registered — overwriting", config.name
            )
        self._clients[config.name] = client_class(config)
        self._connect_locks[config.name] = asyncio.Lock()
        logger.debug("Registered MCP server '%s'", config.name)

    def unregister(self, server_name: str) -> None:
        """Remove a server from the registry (disconnects first if needed)."""
        self._clients.pop(server_name, None)
        self._connect_locks.pop(server_name, None)
        self._capabilities_cache.pop(server_name, None)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect_all(self) -> Dict[str, bool]:
        """
        Attempt to connect all registered servers concurrently.
        Returns {server_name: connected} mapping.
        """
        results = await asyncio.gather(
            *[self._safe_connect(name) for name in self._clients],
            return_exceptions=False,
        )
        return dict(zip(self._clients.keys(), results))

    async def disconnect_all(self) -> None:
        """Disconnect all servers concurrently."""
        await asyncio.gather(
            *[client.disconnect() for client in self._clients.values()],
            return_exceptions=True,
        )
        self._capabilities_cache.clear()

    async def ensure_connected(self, server_name: str) -> bool:
        """Lazy-connect a single server if not already connected."""
        client = self._clients.get(server_name)
        if client is None:
            return False
        if client.is_connected:
            return True
        return await self._safe_connect(server_name)

    async def _safe_connect(self, server_name: str) -> bool:
        async with self._connect_locks[server_name]:
            client = self._clients[server_name]
            if client.is_connected:
                return True
            return await client.connect()

    # ------------------------------------------------------------------
    # Capability discovery
    # ------------------------------------------------------------------

    async def list_all_capabilities(self) -> List[MCPCapabilities]:
        """
        Fetch capabilities from all connected servers.
        Returns list (may include unavailable entries).
        """
        caps = await asyncio.gather(
            *[
                self._get_capabilities(name)
                for name in self._clients
            ],
            return_exceptions=False,
        )
        return list(caps)

    async def list_all_tools(self) -> List[MCPTool]:
        """Flat list of every tool across all connected servers."""
        all_caps = await self.list_all_capabilities()
        tools: List[MCPTool] = []
        for cap in all_caps:
            if cap.available:
                tools.extend(cap.tools)
        return tools

    async def _get_capabilities(self, server_name: str) -> MCPCapabilities:
        if server_name in self._capabilities_cache:
            return self._capabilities_cache[server_name]

        client = self._clients.get(server_name)
        if client is None or not client.is_connected:
            from backend.mcp.client import MCPCapabilities
            return MCPCapabilities(server_name=server_name, available=False)

        caps = await client.list_capabilities()
        if caps.available:
            self._capabilities_cache[server_name] = caps
        return caps

    def invalidate_cache(self, server_name: Optional[str] = None) -> None:
        """Clear cached capabilities for one or all servers."""
        if server_name:
            self._capabilities_cache.pop(server_name, None)
        else:
            self._capabilities_cache.clear()

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    async def call_tool(
        self,
        qualified_name: str,
        arguments: Dict,
    ) -> ToolResult:
        """
        Call a tool by its qualified name: "<server_name>__<tool_name>".

        Lazy-connects the server if not already connected.
        """
        server_name, tool_name = self._parse_tool_name(qualified_name)
        if server_name is None:
            return ToolResult(
                tool_name=qualified_name,
                success=False,
                output="",
                error=f"Invalid tool name '{qualified_name}' — expected 'server__tool'",
            )

        # Lazy connect
        connected = await self.ensure_connected(server_name)
        if not connected:
            return ToolResult(
                tool_name=qualified_name,
                success=False,
                output="",
                error=f"MCP server '{server_name}' is not available",
            )

        client = self._clients[server_name]
        return await client.call_tool(tool_name, arguments)

    async def read_resource(self, server_name: str, uri: str) -> str:
        """Read a resource from a specific server."""
        connected = await self.ensure_connected(server_name)
        if not connected:
            return ""
        client = self._clients[server_name]
        return await client.read_resource(uri)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def registered_servers(self) -> List[str]:
        return list(self._clients.keys())

    @property
    def connected_servers(self) -> List[str]:
        return [
            name
            for name, client in self._clients.items()
            if client.is_connected
        ]

    def is_server_registered(self, server_name: str) -> bool:
        return server_name in self._clients

    def is_server_connected(self, server_name: str) -> bool:
        client = self._clients.get(server_name)
        return client is not None and client.is_connected

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tool_name(qualified_name: str):
        """
        Split "server__tool" into ("server", "tool").
        Returns (None, None) on malformed input.
        """
        parts = qualified_name.split("__", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return None, None
        return parts[0], parts[1]