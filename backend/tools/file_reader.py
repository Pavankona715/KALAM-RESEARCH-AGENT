"""
File Reader Tool
================
Reads content from local files within the allowed upload directory.

Security: Files are restricted to the uploads/ directory only.
Path traversal attacks (../../etc/passwd) are blocked by resolving
the real path and verifying it starts with the allowed base path.

Supports: txt, md, json, csv, yaml, html
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.tools.base import BaseTool
from backend.tools.registry import get_tool_registry
from backend.observability.logger import get_logger

logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".yaml", ".yml", ".html", ".xml"}
MAX_FILE_SIZE_MB = 10


class FileReaderTool(BaseTool):
    """
    Read the contents of uploaded files.

    Best for:
    - Reading documents that have been uploaded by users
    - Accessing configuration or data files
    - Processing text files for analysis
    """

    name = "file_reader"
    description = (
        "Read the contents of a file from the uploads directory. "
        "Supports text files: txt, md, json, csv, yaml, html. "
        "Use the filename as returned by the upload endpoint."
    )
    parameters = {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Name of the file to read (e.g., 'report.txt'). "
                               "Must be in the uploads directory.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters to return. Default: 5000.",
                "default": 5000,
            },
        },
        "required": ["filename"],
    }

    def __init__(self):
        from backend.config.settings import get_settings
        self._upload_dir = Path(get_settings().upload_dir).resolve()

    async def _execute(self, filename: str, max_chars: int = 5000) -> str:
        """Read file with path traversal protection."""

        # Resolve the full path
        file_path = (self._upload_dir / filename).resolve()

        # Security check: ensure file is within uploads directory
        if not str(file_path).startswith(str(self._upload_dir)):
            raise PermissionError(
                f"Access denied: '{filename}' is outside the uploads directory"
            )

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: '{filename}'")

        if not file_path.is_file():
            raise ValueError(f"'{filename}' is not a file")

        # Check extension
        if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"File type '{file_path.suffix}' not supported. "
                f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        # Check size
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(
                f"File too large ({size_mb:.1f}MB). Maximum: {MAX_FILE_SIZE_MB}MB"
            )

        content = file_path.read_text(encoding="utf-8", errors="replace")

        if len(content) > max_chars:
            content = content[:max_chars] + f"\n\n[File truncated at {max_chars} chars]"

        return f"File: {filename}\nSize: {size_mb:.2f}MB\n\n{content}"


# ─── Auto-registration ────────────────────────────────────────────────────────

get_tool_registry().register(FileReaderTool())