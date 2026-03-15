"""
Web Search Tool
===============
Searches the web using Tavily Search API.
Returns a formatted list of results with titles, URLs, and content snippets.

Tavily is purpose-built for AI agents — it returns clean, relevant results
without ads or SEO spam. Alternative: SerpAPI (set SERPAPI_API_KEY instead).

Requires: TAVILY_API_KEY in environment
Install:  pip install tavily-python
"""

from __future__ import annotations

from typing import Any

from backend.tools.base import BaseTool
from backend.tools.registry import get_tool_registry
from backend.observability.logger import get_logger

logger = get_logger(__name__)


class WebSearchTool(BaseTool):
    """
    Search the web for current information.

    Best for:
    - Current events and news
    - Facts that may have changed recently
    - Research on specific topics
    - Finding URLs and sources
    """

    name = "web_search"
    description = (
        "Search the web for current information. Use this when you need "
        "up-to-date facts, news, or information not in your training data. "
        "Returns titles, URLs, and content snippets from top results."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Be specific for better results.",
            },
            "max_results": {
                "type": "integer",
                "description": "Number of results to return (1-10). Default: 5.",
                "default": 5,
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["query"],
    }

    def __init__(self):
        self._client = None  # Lazy initialization

    def _get_client(self):
        """Lazy-load Tavily client to avoid import errors if not installed."""
        if self._client is None:
            try:
                from tavily import TavilyClient
                from backend.config.settings import get_settings
                settings = get_settings()
                if not settings.tavily_api_key:
                    raise ValueError("TAVILY_API_KEY not configured")
                self._client = TavilyClient(api_key=settings.tavily_api_key)
            except ImportError:
                raise ImportError(
                    "tavily-python not installed. Run: pip install tavily-python"
                )
        return self._client

    async def _execute(self, query: str, max_results: int = 5) -> str:
        """Execute web search and format results."""
        import asyncio

        client = self._get_client()

        # Tavily is synchronous — run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_answer=True,  # Get AI-synthesized answer too
            )
        )

        return self._format_results(query, response)

    def _format_results(self, query: str, response: dict) -> str:
        """Format Tavily response into a clean string for the LLM."""
        lines = [f"Search results for: '{query}'\n"]

        # Include the synthesized answer if available
        if response.get("answer"):
            lines.append(f"Summary: {response['answer']}\n")

        results = response.get("results", [])
        if not results:
            return f"No results found for: '{query}'"

        lines.append(f"Top {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            lines.append(f"{i}. {result.get('title', 'No title')}")
            lines.append(f"   URL: {result.get('url', '')}")
            content = result.get("content", "")
            if content:
                # Truncate long snippets
                snippet = content[:400] + "..." if len(content) > 400 else content
                lines.append(f"   {snippet}")
            lines.append("")

        return "\n".join(lines)


class WebSearchToolStub(BaseTool):
    """
    Stub web search for testing without a Tavily API key.
    Returns clearly marked fake results.
    """

    name = "web_search"
    description = WebSearchTool.description
    parameters = WebSearchTool.parameters

    async def _execute(self, query: str, max_results: int = 5) -> str:
        return (
            f"[STUB] Web search results for: '{query}'\n\n"
            f"1. Example Result 1\n"
            f"   URL: https://example.com/result1\n"
            f"   This is a stub result. Configure TAVILY_API_KEY for real search.\n\n"
            f"2. Example Result 2\n"
            f"   URL: https://example.com/result2\n"
            f"   Another stub result for testing purposes.\n"
        )


# ─── Auto-registration ────────────────────────────────────────────────────────

def _register():
    """Register the appropriate web search tool based on configuration."""
    from backend.config.settings import get_settings
    registry = get_tool_registry()
    settings = get_settings()

    if settings.tavily_api_key:
        registry.register(WebSearchTool())
        logger.info("web_search_registered", provider="tavily")
    else:
        registry.register(WebSearchToolStub())
        logger.info("web_search_registered", provider="stub", note="Set TAVILY_API_KEY for real search")


_register()