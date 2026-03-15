"""
PDF Parser
==========
Extracts text from PDF files using pypdf.
Falls back to page-by-page extraction if whole-document fails.
"""

from __future__ import annotations

from pathlib import Path

from backend.observability.logger import get_logger

logger = get_logger(__name__)


async def parse_pdf(file_path: Path) -> str:
    """
    Extract all text from a PDF file.
    Returns the full text content as a single string.
    """
    try:
        import pypdf
    except ImportError:
        raise ImportError("pypdf not installed. Run: pip install pypdf")

    import asyncio

    def _extract() -> str:
        reader = pypdf.PdfReader(str(file_path))
        pages = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(text)
            except Exception as e:
                logger.warning("pdf_page_extraction_failed", page=i, error=str(e))
        return "\n\n".join(pages)

    # Run blocking pypdf in thread pool
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(None, _extract)

    logger.debug("pdf_parsed", path=str(file_path), chars=len(text))
    return text