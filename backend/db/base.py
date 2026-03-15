"""
SQLAlchemy Base Model
=====================
Declarative base and reusable mixins for all database models.

Every model gets:
- id: UUID primary key (better than auto-increment for distributed systems)
- created_at / updated_at: automatic timestamps
- to_dict(): serialization helper
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


class TimestampMixin:
    """
    Adds created_at and updated_at to any model.
    updated_at is automatically set on every update via onupdate.
    """
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class UUIDMixin:
    """UUID primary key — preferred over integer IDs for distributed systems."""
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )


class BaseModel(UUIDMixin, TimestampMixin, Base):
    """
    Abstract base combining UUID PK + timestamps.
    Use this as the base for all domain models.

    Usage:
        class User(BaseModel):
            __tablename__ = "users"
            email: Mapped[str] = mapped_column(String(255), unique=True)
    """
    __abstract__ = True

    def to_dict(self) -> dict:
        """Convert model to dictionary (excludes SQLAlchemy internals)."""
        return {
            col.name: getattr(self, col.name)
            for col in self.__table__.columns
        }