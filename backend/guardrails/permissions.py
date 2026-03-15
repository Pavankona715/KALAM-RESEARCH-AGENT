"""
Tool permission enforcement per user role.

Design rationale:
- Uses the same Role hierarchy established in db/models/user.py
- Permissions are data-driven (dict) not code-branching, making it trivial
  to add new roles or tools without touching agent logic
- PermissionChecker is injected via Depends() so tests can mock it
- Three permission levels for tools: ALLOWED, DENIED, RATE_LIMITED
- Rate limiting is per-user per-tool with Redis (fallback: in-memory counter)
- Agent types inherit a tool allowlist; user role further restricts it

Permission matrix (tool → minimum required role):
    free:    web_search, wikipedia, calculator
    basic:   + file_reader, database_query(read-only)
    pro:     + all tools
    admin:   all tools + MCP connectors

MCP connectors are always restricted to pro/admin.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Role hierarchy
# ---------------------------------------------------------------------------


class UserRole(str, Enum):
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ADMIN = "admin"

    @property
    def level(self) -> int:
        return {"free": 0, "basic": 1, "pro": 2, "admin": 3}[self.value]

    def has_at_least(self, required: "UserRole") -> bool:
        return self.level >= required.level


# ---------------------------------------------------------------------------
# Permission definitions
# ---------------------------------------------------------------------------

# Tool name → minimum role required to use it
TOOL_MINIMUM_ROLES: Dict[str, UserRole] = {
    # Available to everyone
    "web_search": UserRole.FREE,
    "wikipedia": UserRole.FREE,
    "calculator": UserRole.FREE,
    # Basic+ tools
    "file_reader": UserRole.BASIC,
    "database_query": UserRole.BASIC,
    # Pro+ tools
    "code_executor": UserRole.PRO,
    "shell_command": UserRole.PRO,
    # MCP connectors — always pro+
    "google_drive__list_files": UserRole.PRO,
    "google_drive__read_file": UserRole.PRO,
    "google_drive__search_files": UserRole.PRO,
    "google_drive__create_file": UserRole.PRO,
    "notion__search_pages": UserRole.PRO,
    "notion__read_page": UserRole.PRO,
    "notion__create_page": UserRole.PRO,
    "notion__append_to_page": UserRole.PRO,
    "notion__list_databases": UserRole.PRO,
    "notion__query_database": UserRole.PRO,
    "slack__list_channels": UserRole.PRO,
    "slack__read_channel": UserRole.PRO,
    "slack__send_message": UserRole.PRO,
    "slack__search_messages": UserRole.PRO,
    "slack__get_thread": UserRole.PRO,
    "slack__list_users": UserRole.PRO,
    # Admin-only
    "admin_db_write": UserRole.ADMIN,
}

# Per-user rate limits: (window_seconds, max_calls)
TOOL_RATE_LIMITS: Dict[str, tuple] = {
    "web_search": (60, 20),       # 20 per minute
    "database_query": (60, 10),
    "code_executor": (60, 5),
    "shell_command": (60, 2),
    # MCP connectors
    "google_drive__create_file": (3600, 50),
    "slack__send_message": (60, 10),
    "notion__create_page": (3600, 20),
}

# Default rate limit applied to all tools not in TOOL_RATE_LIMITS
_DEFAULT_RATE_LIMIT = (60, 100)  # 100 calls per minute


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class PermissionDecision(str, Enum):
    ALLOWED = "allowed"
    DENIED_ROLE = "denied_role"
    DENIED_RATE_LIMIT = "denied_rate_limit"
    DENIED_UNKNOWN_TOOL = "denied_unknown_tool"


@dataclass
class PermissionResult:
    decision: PermissionDecision
    tool_name: str
    user_id: str
    user_role: str
    reason: Optional[str] = None
    retry_after_seconds: Optional[int] = None

    @property
    def allowed(self) -> bool:
        return self.decision == PermissionDecision.ALLOWED


# ---------------------------------------------------------------------------
# In-memory rate limiter (fallback when Redis unavailable)
# ---------------------------------------------------------------------------


class _InMemoryRateLimiter:
    """Sliding-window rate limiter backed by a simple dict."""

    def __init__(self) -> None:
        # {user_id:tool_name → list of call timestamps}
        self._windows: Dict[str, List[float]] = {}

    def is_allowed(self, key: str, window_seconds: int, max_calls: int) -> bool:
        now = time.time()
        window_start = now - window_seconds
        calls = self._windows.get(key, [])
        # Evict old entries
        calls = [t for t in calls if t > window_start]
        self._windows[key] = calls
        if len(calls) >= max_calls:
            return False
        calls.append(now)
        return True

    def time_until_reset(self, key: str, window_seconds: int) -> int:
        calls = self._windows.get(key, [])
        if not calls:
            return 0
        oldest = min(calls)
        return max(0, int(window_seconds - (time.time() - oldest)))


# ---------------------------------------------------------------------------
# PermissionChecker
# ---------------------------------------------------------------------------


class PermissionChecker:
    """
    Enforces tool access based on user role and rate limits.

    Injected via FastAPI Depends() — redis_client is optional.
    """

    def __init__(self, redis_client=None) -> None:
        self._redis = redis_client
        self._fallback_limiter = _InMemoryRateLimiter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check(
        self,
        tool_name: str,
        user_id: str,
        user_role: str,
    ) -> PermissionResult:
        """
        Check if user_role is permitted to call tool_name.

        Returns PermissionResult — never raises.
        """
        role = self._parse_role(user_role)

        # 1. Role check
        minimum_role = TOOL_MINIMUM_ROLES.get(tool_name)
        if minimum_role is None:
            # Tool not in permission matrix → deny unknown tools
            return PermissionResult(
                decision=PermissionDecision.DENIED_UNKNOWN_TOOL,
                tool_name=tool_name,
                user_id=user_id,
                user_role=user_role,
                reason=f"Tool '{tool_name}' is not registered in the permission matrix",
            )

        if not role.has_at_least(minimum_role):
            return PermissionResult(
                decision=PermissionDecision.DENIED_ROLE,
                tool_name=tool_name,
                user_id=user_id,
                user_role=user_role,
                reason=f"Tool '{tool_name}' requires role '{minimum_role.value}' (current: '{user_role}')",
            )

        # 2. Rate limit check
        rate_result = await self._check_rate_limit(tool_name, user_id)
        if not rate_result["allowed"]:
            return PermissionResult(
                decision=PermissionDecision.DENIED_RATE_LIMIT,
                tool_name=tool_name,
                user_id=user_id,
                user_role=user_role,
                reason=f"Rate limit exceeded for tool '{tool_name}'",
                retry_after_seconds=rate_result.get("retry_after"),
            )

        return PermissionResult(
            decision=PermissionDecision.ALLOWED,
            tool_name=tool_name,
            user_id=user_id,
            user_role=user_role,
        )

    def get_allowed_tools(self, user_role: str) -> Set[str]:
        """Return the set of tool names accessible to a given role."""
        role = self._parse_role(user_role)
        return {
            tool
            for tool, min_role in TOOL_MINIMUM_ROLES.items()
            if role.has_at_least(min_role)
        }

    def filter_tools(self, tool_names: List[str], user_role: str) -> List[str]:
        """Return only the tool names the user is allowed to use."""
        allowed = self.get_allowed_tools(user_role)
        return [t for t in tool_names if t in allowed]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_role(user_role: str) -> UserRole:
        try:
            return UserRole(user_role.lower())
        except ValueError:
            logger.warning("Unknown user role '%s' — defaulting to FREE", user_role)
            return UserRole.FREE

    async def _check_rate_limit(
        self, tool_name: str, user_id: str
    ) -> Dict:
        window_seconds, max_calls = TOOL_RATE_LIMITS.get(
            tool_name, _DEFAULT_RATE_LIMIT
        )
        key = f"rate:{user_id}:{tool_name}"

        if self._redis:
            try:
                return await self._redis_rate_check(key, window_seconds, max_calls)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Redis rate limit check failed, using fallback: %s", exc)

        # Fallback to in-memory
        allowed = self._fallback_limiter.is_allowed(key, window_seconds, max_calls)
        retry = (
            self._fallback_limiter.time_until_reset(key, window_seconds)
            if not allowed
            else None
        )
        return {"allowed": allowed, "retry_after": retry}

    async def _redis_rate_check(
        self, key: str, window_seconds: int, max_calls: int
    ) -> Dict:
        """Sliding window rate limit using Redis sorted sets."""
        now = time.time()
        window_start = now - window_seconds

        pipe = self._redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, window_seconds * 2)
        results = await pipe.execute()

        current_count = results[1]  # count BEFORE this call
        if current_count >= max_calls:
            # Undo the zadd we just did
            await self._redis.zrem(key, str(now))
            oldest_score = await self._redis.zrange(key, 0, 0, withscores=True)
            retry = 0
            if oldest_score:
                retry = max(0, int(window_seconds - (now - oldest_score[0][1])))
            return {"allowed": False, "retry_after": retry}

        return {"allowed": True}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_checker_instance: Optional[PermissionChecker] = None


def get_permission_checker(redis_client=None) -> PermissionChecker:
    """FastAPI dependency."""
    global _checker_instance
    if _checker_instance is None:
        _checker_instance = PermissionChecker(redis_client=redis_client)
    return _checker_instance