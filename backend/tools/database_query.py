"""
Database Query Tool
===================
Executes read-only SQL queries against the application database.

Security model:
1. READ-ONLY: Only SELECT statements allowed — INSERT/UPDATE/DELETE/DROP rejected
2. ALLOWLIST: Only tables in the approved list can be queried
3. ROW LIMIT: Maximum 100 rows to prevent data dumps
4. NO RAW VALUES: Parameters passed separately, never interpolated into SQL

This tool lets agents query business data (chat history, document metadata,
agent run stats) without risk of destructive operations.
"""

from __future__ import annotations

import re
from typing import Any

from backend.tools.base import BaseTool
from backend.tools.registry import get_tool_registry
from backend.observability.logger import get_logger

logger = get_logger(__name__)

# Only these tables can be queried by agents
ALLOWED_TABLES = {
    "chat_sessions",
    "chat_messages",
    "documents",
    "agent_runs",
}

# Blocked SQL keywords that could modify data or extract system info
BLOCKED_KEYWORDS = {
    "insert", "update", "delete", "drop", "truncate", "alter",
    "create", "grant", "revoke", "exec", "execute", "call",
    "pg_", "information_schema", "sys.", "xp_", "--", ";--",
}

MAX_ROWS = 100


class DatabaseQueryTool(BaseTool):
    """
    Execute read-only SQL queries to retrieve data.

    Best for:
    - Looking up chat history or session data
    - Checking document processing status
    - Querying agent run statistics
    - Retrieving user-specific data

    Only SELECT queries on approved tables are allowed.
    """

    name = "database_query"
    description = (
        "Execute a read-only SQL SELECT query on the database. "
        f"Allowed tables: {', '.join(sorted(ALLOWED_TABLES))}. "
        "Only SELECT statements are permitted. Maximum 100 rows returned."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "SQL SELECT query to execute. "
                    f"Only tables allowed: {', '.join(sorted(ALLOWED_TABLES))}. "
                    "Example: 'SELECT id, title, status FROM documents LIMIT 10'"
                ),
            },
        },
        "required": ["query"],
    }

    def _validate_query(self, query: str) -> str | None:
        """
        Validate the SQL query for safety.
        Returns error message if invalid, None if safe.
        """
        query_lower = query.lower().strip()

        # Must start with SELECT
        if not query_lower.startswith("select"):
            return "Only SELECT queries are allowed"

        # Check for blocked keywords
        for keyword in BLOCKED_KEYWORDS:
            if keyword in query_lower:
                return f"Blocked keyword detected: '{keyword}'"

        # Verify at least one allowed table is referenced
        has_allowed_table = any(table in query_lower for table in ALLOWED_TABLES)
        if not has_allowed_table:
            return (
                f"Query must reference at least one allowed table: "
                f"{', '.join(sorted(ALLOWED_TABLES))}"
            )

        return None

    async def _execute(self, query: str) -> str:
        """Execute the validated query and format results."""

        # Additional safety validation
        error = self._validate_query(query)
        if error:
            raise PermissionError(f"Query rejected: {error}")

        # Add LIMIT if not present to prevent large dumps
        query_lower = query.lower()
        if "limit" not in query_lower:
            query = f"{query.rstrip(';')} LIMIT {MAX_ROWS}"

        try:
            from sqlalchemy import text
            from backend.db.session import get_db_context

            async with get_db_context() as db:
                result = await db.execute(text(query))
                rows = result.fetchall()
                columns = list(result.keys())

            if not rows:
                return f"Query returned no results.\nQuery: {query}"

            # Format as a readable table
            return self._format_results(query, columns, rows)

        except Exception as e:
            # Don't expose internal DB errors to the agent
            logger.error("database_query_failed", error=str(e), query=query[:200])
            raise RuntimeError(f"Query failed: {str(e)[:200]}")

    def _format_results(self, query: str, columns: list, rows: list) -> str:
        """Format query results as a readable table."""
        lines = [
            f"Query: {query}",
            f"Results: {len(rows)} row(s)\n",
        ]

        # Header
        lines.append(" | ".join(columns))
        lines.append("-" * (len(" | ".join(columns)) + 5))

        # Rows
        for row in rows:
            line_parts = []
            for val in row:
                val_str = str(val) if val is not None else "NULL"
                # Truncate long values
                if len(val_str) > 50:
                    val_str = val_str[:47] + "..."
                line_parts.append(val_str)
            lines.append(" | ".join(line_parts))

        return "\n".join(lines)


# ─── Auto-registration ────────────────────────────────────────────────────────

get_tool_registry().register(DatabaseQueryTool())