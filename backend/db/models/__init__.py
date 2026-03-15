"""
Database Models
===============
Import all models here so SQLAlchemy and Alembic can discover them
when creating/running migrations.
"""

from backend.db.base import Base, BaseModel
from backend.db.models.user import User
from backend.db.models.chat import ChatSession, ChatMessage, MessageRole
from backend.db.models.document import Document, DocumentStatus, DocumentType
from backend.db.models.agent_run import AgentRun, AgentRunStatus

__all__ = [
    "Base",
    "BaseModel",
    "User",
    "ChatSession",
    "ChatMessage",
    "MessageRole",
    "Document",
    "DocumentStatus",
    "DocumentType",
    "AgentRun",
    "AgentRunStatus",
]