"""
FastAPI Dependencies
====================
Reusable dependencies injected into route handlers via Depends().

Design: Never import settings or db sessions directly in routes.
Always go through these dependencies — it makes testing trivial
(just override the dependency in the test client).
"""

from typing import Annotated, Optional
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.settings import AppSettings, get_settings
from backend.db.session import get_db_session
from backend.db.models.user import User
from backend.db.repositories.user_repo import user_repo
from backend.observability.logger import get_logger

logger = get_logger(__name__)


# ─── Type aliases (clean up route signatures) ─────────────────────────────────

DBSession = Annotated[AsyncSession, Depends(get_db_session)]
Settings = Annotated[AppSettings, Depends(get_settings)]


# ─── Authentication ────────────────────────────────────────────────────────────

async def get_current_user(
    db: DBSession,
    x_api_key: Annotated[Optional[str], Header()] = None,
) -> User:
    """
    Authenticate request via API key in X-Api-Key header.

    Usage in routes:
        @router.get("/protected")
        async def protected(user: CurrentUser):
            return {"user_id": str(user.id)}
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-Api-Key header required",
        )

    user = await user_repo.get_by_api_key(db, x_api_key)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key",
        )

    return user


async def get_optional_user(
    db: DBSession,
    x_api_key: Annotated[Optional[str], Header()] = None,
) -> User | None:
    """Optional auth — returns None if no API key provided."""
    if not x_api_key:
        return None
    return await user_repo.get_by_api_key(db, x_api_key)


CurrentUser = Annotated[User, Depends(get_current_user)]
OptionalUser = Annotated[User | None, Depends(get_optional_user)]


# ─── Pagination ────────────────────────────────────────────────────────────────

class PaginationParams:
    """Standard pagination query parameters."""

    def __init__(
        self,
        skip: int = 0,
        limit: int = 20,
    ):
        if skip < 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="skip must be >= 0",
            )
        if limit < 1 or limit > 100:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="limit must be between 1 and 100",
            )
        self.skip = skip
        self.limit = limit


Pagination = Annotated[PaginationParams, Depends(PaginationParams)]