"""
MCP (Model Context Protocol) base client.

Design rationale:
- MCPClient is a Protocol so business logic never imports concrete transports
- All methods are async — MCP servers are always remote or subprocess-based
- Graceful degradation: if a server is unavailable, capabilities return empty
- Tool results map to our existing ToolResult schema so agents need no changes
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from backend.tools.base import ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class MCPTransport(str, Enum):
    """Supported MCP transport mechanisms."""
    STDIO = "stdio"
    SSE = "sse"
    HTTP = "http"


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""
    name: str
    transport: MCPTransport
    # For SSE / HTTP transports
    url: Optional[str] = None
    # For stdio transport
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    # Optional auth headers / env vars
    env: Dict[str, str] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    enabled: bool = True


@dataclass
class MCPTool:
    """A tool exposed by an MCP server."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": f"{self.server_name}__{self.name}",
            "description": self.description,
            "parameters": self.input_schema,
        }


@dataclass
class MCPResource:
    """A resource (file, note, etc.) exposed by an MCP server."""
    uri: str
    name: str
    description: str
    mime_type: str
    server_name: str


@dataclass
class MCPCapabilities:
    """Describes what a connected MCP server can do."""
    tools: List[MCPTool] = field(default_factory=list)
    resources: List[MCPResource] = field(default_factory=list)
    prompts: List[Dict[str, Any]] = field(default_factory=list)
    server_name: str = ""
    server_version: str = ""
    available: bool = False


# ---------------------------------------------------------------------------
# Protocol interface
# ---------------------------------------------------------------------------


@runtime_checkable
class MCPClientProtocol(Protocol):
    """Protocol that all MCP client implementations must satisfy."""

    @property
    def server_name(self) -> str: ...

    @property
    def is_connected(self) -> bool: ...

    async def connect(self) -> bool: ...

    async def disconnect(self) -> None: ...

    async def list_capabilities(self) -> MCPCapabilities: ...

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> ToolResult: ...

    async def read_resource(self, uri: str) -> str: ...


# ---------------------------------------------------------------------------
# Base implementation (ABC)
# ---------------------------------------------------------------------------


class BaseMCPClient(ABC):
    """
    Abstract base for MCP clients.

    Concrete connectors (GoogleDrive, Notion, Slack, etc.) extend this.
    Each connector overrides _connect_impl / _call_tool_impl / etc.
    """

    def __init__(self, config: MCPServerConfig) -> None:
        self._config = config
        self._connected: bool = False
        self._capabilities: Optional[MCPCapabilities] = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def server_name(self) -> str:
        return self._config.name

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Public lifecycle methods
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """
        Establish connection to the MCP server.
        Returns True on success, False on failure (never raises).
        """
        if not self._config.enabled:
            logger.info("MCP server '%s' is disabled — skipping", self.server_name)
            return False

        async with self._lock:
            if self._connected:
                return True
            try:
                self._connected = await self._connect_impl()
                if self._connected:
                    logger.info("MCP server '%s' connected", self.server_name)
                else:
                    logger.warning(
                        "MCP server '%s' connect returned False", self.server_name
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "MCP server '%s' failed to connect: %s", self.server_name, exc
                )
                self._connected = False
        return self._connected

    async def disconnect(self) -> None:
        """Gracefully close the connection."""
        async with self._lock:
            if self._connected:
                try:
                    await self._disconnect_impl()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "MCP server '%s' disconnect error: %s", self.server_name, exc
                    )
                finally:
                    self._connected = False
                    self._capabilities = None

    async def list_capabilities(self) -> MCPCapabilities:
        """
        Return capabilities, fetching from server if needed.
        Returns empty MCPCapabilities (available=False) if not connected.
        """
        if not self._connected:
            return MCPCapabilities(server_name=self.server_name, available=False)

        if self._capabilities is None:
            try:
                self._capabilities = await self._fetch_capabilities()
                self._capabilities.available = True
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "MCP server '%s' capabilities fetch failed: %s",
                    self.server_name,
                    exc,
                )
                self._capabilities = MCPCapabilities(
                    server_name=self.server_name, available=False
                )
        return self._capabilities

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> ToolResult:
        """
        Invoke a tool on the MCP server.
        Returns a ToolResult with success=False if not connected or on error.
        """
        if not self._connected:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"MCP server '{self.server_name}' is not connected",
            )
        try:
            return await self._call_tool_impl(tool_name, arguments)
        except asyncio.TimeoutError:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"MCP tool '{tool_name}' timed out after {self._config.timeout}s",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "MCP server '%s' tool '%s' error: %s", self.server_name, tool_name, exc
            )
            return ToolResult(tool_name=tool_name, success=False, output="", error=str(exc))

    async def read_resource(self, uri: str) -> str:
        """
        Read a resource by URI.
        Returns empty string if not connected or on error.
        """
        if not self._connected:
            return ""
        try:
            return await self._read_resource_impl(uri)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "MCP server '%s' resource '%s' error: %s", self.server_name, uri, exc
            )
            return ""

    # ------------------------------------------------------------------
    # Abstract implementation hooks (override in subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    async def _connect_impl(self) -> bool:
        """Perform the actual connection. Return True on success."""

    async def _disconnect_impl(self) -> None:
        """Perform graceful teardown. Override if needed."""

    @abstractmethod
    async def _fetch_capabilities(self) -> MCPCapabilities:
        """Fetch tools/resources from the connected server."""

    @abstractmethod
    async def _call_tool_impl(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> ToolResult:
        """Invoke a tool. Called only when connected."""

    async def _read_resource_impl(self, uri: str) -> str:
        """Read a resource. Override in connectors that support resources."""
        return ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_headers(self) -> Dict[str, str]:
        """Merge config headers with content-type defaults."""
        return {"Content-Type": "application/json", **self._config.headers}

    def _parse_tool_result(self, raw: Any) -> str:
        """Normalize tool result payload to a string."""
        if isinstance(raw, str):
            return raw
        if isinstance(raw, (dict, list)):
            return json.dumps(raw, ensure_ascii=False, indent=2)
        return str(raw)