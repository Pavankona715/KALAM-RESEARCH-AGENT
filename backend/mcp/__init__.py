"""
MCP (Model Context Protocol) package.

Provides:
- BaseMCPClient / MCPClientProtocol for implementing connectors
- MCPRegistry for managing a pool of clients
- get_mcp_registry() FastAPI dependency
- setup_mcp_registry() helper for lifespan startup
"""

from backend.mcp.client import (
    BaseMCPClient,
    MCPCapabilities,
    MCPClientProtocol,
    MCPResource,
    MCPServerConfig,
    MCPTool,
    MCPTransport,
)
from backend.mcp.registry import MCPRegistry, get_mcp_registry

__all__ = [
    "BaseMCPClient",
    "MCPCapabilities",
    "MCPClientProtocol",
    "MCPRegistry",
    "MCPResource",
    "MCPServerConfig",
    "MCPTool",
    "MCPTransport",
    "get_mcp_registry",
    "setup_mcp_registry",
]


async def setup_mcp_registry() -> MCPRegistry:
    """
    Called during FastAPI lifespan startup.

    Reads environment variables and registers any configured MCP servers.
    Servers without required env vars are silently skipped.
    """
    import logging

    from backend.mcp.connectors.google_drive import (
        GoogleDriveMCPClient,
        build_google_drive_config,
    )
    from backend.mcp.connectors.notion import NotionMCPClient, build_notion_config
    from backend.mcp.connectors.slack import SlackMCPClient, build_slack_config

    log = logging.getLogger(__name__)
    registry = get_mcp_registry()

    # Google Drive
    gd_config = build_google_drive_config()
    if gd_config.enabled:
        registry.register(gd_config, GoogleDriveMCPClient)
        log.info("Registered MCP connector: google_drive")

    # Notion
    notion_config = build_notion_config()
    if notion_config.enabled:
        registry.register(notion_config, NotionMCPClient)
        log.info("Registered MCP connector: notion")

    # Slack
    slack_config = build_slack_config()
    if slack_config.enabled:
        registry.register(slack_config, SlackMCPClient)
        log.info("Registered MCP connector: slack")

    # Connect all registered servers concurrently
    if registry.registered_servers:
        results = await registry.connect_all()
        for name, connected in results.items():
            status = "connected" if connected else "FAILED"
            log.info("MCP server '%s': %s", name, status)
    else:
        log.info("No MCP servers configured — MCP layer disabled")

    return registry