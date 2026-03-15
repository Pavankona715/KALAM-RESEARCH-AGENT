"""
Document Model
==============
Tracks uploaded/ingested documents and their processing status.
The actual content lives in Qdrant (vectors) and the file system.
PostgreSQL stores metadata, ownership, and ingestion status.
"""

import uuid
from enum import Enum as PyEnum

from sqlalchemy import BigInteger, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import BaseModel


class DocumentStatus(str, PyEnum):
    PENDING = "pending"        # Uploaded, not yet processed
    PROCESSING = "processing"  # Currently being chunked/embedded
    READY = "ready"            # Fully ingested, available for RAG
    FAILED = "failed"          # Ingestion failed


class DocumentType(str, PyEnum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"
    HTML = "html"
    URL = "url"


class Document(BaseModel):
    __tablename__ = "documents"

    # Ownership
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # File metadata
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    doc_type: Mapped[str] = mapped_column(String(20), nullable=False)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    file_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(2000), nullable=True)

    # Content
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_preview: Mapped[str | None] = mapped_column(
        String(500), nullable=True
    )  # First 500 chars for display

    # Processing status
    status: Mapped[str] = mapped_column(
        String(20), default=DocumentStatus.PENDING, nullable=False, index=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Ingestion stats
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Vector store reference
    qdrant_collection: Mapped[str | None] = mapped_column(String(200), nullable=True)

    # Extensible metadata (tags, custom fields)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)

    # Relationships
    owner: Mapped["User"] = relationship(back_populates="documents")  # noqa: F821

    def __repr__(self) -> str:
        return (
            f"<Document id={self.id} "
            f"filename={self.original_filename} "
            f"status={self.status}>"
        )