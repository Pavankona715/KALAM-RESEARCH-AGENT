"""
Wikipedia Tool
==============
Fetches article summaries from Wikipedia.

Uses the wikipedia-api library which handles disambiguation,
redirects, and rate limiting automatically.

Best for: factual lookups, definitions, historical information,
          concept explanations — things unlikely to change frequently.

Requires: pip install wikipedia-api
"""

from __future__ import annotations

from typing import Any

from backend.tools.base import BaseTool
from backend.tools.registry import get_tool_registry
from backend.observability.logger import get_logger

logger = get_logger(__name__)


class WikipediaTool(BaseTool):
    """
    Look up information on Wikipedia.

    Best for:
    - Definitions and explanations of concepts
    - Historical facts and dates
    - Biographies of notable people
    - Information about places, organizations, scientific topics
    """

    name = "wikipedia"
    description = (
        "Look up information on Wikipedia. Best for factual information, "
        "definitions, historical facts, and concept explanations. "
        "Returns a summary of the most relevant Wikipedia article."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Topic or concept to look up on Wikipedia. Use short names e.g. 'APJ Abdul Kalam' not 'Who is APJ Abdul Kalam'.",
            },
            "sentences": {
                "type": "integer",
                "description": "Number of summary sentences to return (1-10). Default: 5.",
                "default": 5,
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["query"],
    }

    def __init__(self):
        self._wiki = None

    def _get_wiki(self):
        if self._wiki is None:
            try:
                import wikipediaapi
                self._wiki = wikipediaapi.Wikipedia(
                    user_agent="UniversalAIAgent/1.0",
                    language="en",
                )
            except ImportError:
                raise ImportError(
                    "wikipedia-api not installed. Run: pip install wikipedia-api"
                )
        return self._wiki

    async def _execute(self, query: str = "", sentences: int = 5, **kwargs) -> str:
        import asyncio

        # Force sentences to int in case llama3.2 sends a string
        try:
            sentences = int(sentences) if sentences else 5
        except (ValueError, TypeError):
            sentences = 5

        # Guard against empty queries
        if not query or not query.strip():
            return "Please provide a search query. Example: 'APJ Abdul Kalam'"

        # Clean up the query — remove question words that confuse Wikipedia lookup
        clean_query = query.strip()
        for prefix in ["who is ", "what is ", "tell me about ", "explain "]:
            if clean_query.lower().startswith(prefix):
                clean_query = clean_query[len(prefix):]
                break

        wiki = self._get_wiki()

        # Run synchronous wikipedia-api in thread pool
        loop = asyncio.get_event_loop()
        page = await loop.run_in_executor(
            None,
            lambda: wiki.page(clean_query)
        )

        if not page.exists():
            # Try search-based fallback
            return await self._search_fallback(wiki, clean_query, sentences)

        summary = page.summary
        if not summary:
            return f"No summary available for '{clean_query}' on Wikipedia."

        # Return first N sentences
        import re
        sentence_list = re.split(r"(?<=[.!?])\s+", summary)
        trimmed = " ".join(sentence_list[:sentences])

        return (
            f"Wikipedia: {page.title}\n"
            f"URL: {page.fullurl}\n\n"
            f"{trimmed}"
        )

    async def _search_fallback(self, wiki, query: str, sentences: int) -> str:
        """Search Wikipedia and return the first matching article."""
        import asyncio

        try:
            import wikipedia

            loop = asyncio.get_event_loop()

            # Search for matching pages
            search_results = await loop.run_in_executor(
                None,
                lambda: wikipedia.search(query, results=3)
            )

            if not search_results:
                return f"No Wikipedia article found for '{query}'."

            # Try the first result
            page = await loop.run_in_executor(
                None,
                lambda: wiki.page(search_results[0])
            )

            if not page.exists():
                return f"No Wikipedia article found for '{query}'."

            import re
            sentence_list = re.split(r"(?<=[.!?])\s+", page.summary)
            trimmed = " ".join(sentence_list[:sentences])

            return (
                f"Wikipedia: {page.title}\n"
                f"URL: {page.fullurl}\n\n"
                f"{trimmed}"
            )

        except Exception as e:
            logger.warning("wikipedia_search_fallback_failed", error=str(e))
            return f"No Wikipedia article found for '{query}'."


# ─── Auto-registration ────────────────────────────────────────────────────────

get_tool_registry().register(WikipediaTool())