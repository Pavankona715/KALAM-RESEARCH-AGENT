"""
URL / HTML Parser
=================
Fetches and extracts clean text from web pages and HTML files.

For URLs:   httpx fetches the page, BeautifulSoup extracts text
For HTML:   BeautifulSoup extracts text directly from the file

Strips: scripts, styles, nav, header, footer, ads
Keeps:  article body, paragraphs, headings, lists
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from backend.observability.logger import get_logger

logger = get_logger(__name__)

# Tags to remove entirely before extracting text
NOISE_TAGS = [
    "script", "style", "nav", "header", "footer",
    "aside", "advertisement", "iframe", "noscript",
]


def _extract_text_from_html(html: str, source: str = "") -> str:
    """Extract clean text from HTML using BeautifulSoup."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("beautifulsoup4 not installed. Run: pip install beautifulsoup4")

    soup = BeautifulSoup(html, "html.parser")

    # Remove noise elements
    for tag in soup(NOISE_TAGS):
        tag.decompose()

    # Try to find the main content area
    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id="content")
        or soup.find(class_="content")
        or soup.find("body")
        or soup
    )

    # Extract text with paragraph spacing
    lines = []
    for element in main_content.find_all(["p", "h1", "h2", "h3", "h4", "li", "td"]):
        text = element.get_text(separator=" ", strip=True)
        if text and len(text) > 20:  # Skip tiny fragments
            lines.append(text)

    return "\n\n".join(lines)


async def parse_html_file(file_path: Path) -> str:
    """Extract text from a local HTML file."""
    import asyncio

    def _read_and_parse() -> str:
        html = file_path.read_text(encoding="utf-8", errors="replace")
        return _extract_text_from_html(html, source=str(file_path))

    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(None, _read_and_parse)

    logger.debug("html_parsed", path=str(file_path), chars=len(text))
    return text


async def parse_url(url: str, timeout: int = 30) -> str:
    """
    Fetch a URL and extract its text content.

    Args:
        url: The web page URL to fetch
        timeout: Request timeout in seconds

    Returns:
        Clean text extracted from the page
    """
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx not installed. Run: pip install httpx")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; UniversalAIAgent/1.0; "
            "+https://github.com/your-repo)"
        )
    }

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ValueError(f"HTTP {e.response.status_code} fetching {url}")
        except httpx.RequestError as e:
            raise ValueError(f"Network error fetching {url}: {e}")

    content_type = response.headers.get("content-type", "")
    if "html" not in content_type and "text" not in content_type:
        raise ValueError(
            f"URL returned non-text content type: {content_type}. "
            "Only HTML and text pages are supported."
        )

    text = _extract_text_from_html(response.text, source=url)

    logger.info("url_parsed", url=url, chars=len(text))
    return text