"""
Text / Markdown Parser
======================
Reads plain text files with encoding detection.
Handles UTF-8, Latin-1, and falls back to replacing bad bytes.
"""

from __future__ import annotations

from pathlib import Path

from backend.observability.logger import get_logger

logger = get_logger(__name__)

ENCODINGS_TO_TRY = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]


async def parse_txt(file_path: Path) -> str:
    """Read a plain text or markdown file, auto-detecting encoding."""
    import asyncio

    def _read() -> str:
        for encoding in ENCODINGS_TO_TRY:
            try:
                text = file_path.read_text(encoding=encoding)
                return text
            except (UnicodeDecodeError, LookupError):
                continue

        # Last resort: replace undecodable bytes
        return file_path.read_text(encoding="utf-8", errors="replace")

    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(None, _read)

    logger.debug("txt_parsed", path=str(file_path), chars=len(text))
    return text