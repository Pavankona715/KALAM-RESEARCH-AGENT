"""
Document Ingestion Pipeline
============================
Orchestrates the full ingestion flow for uploaded documents:

  File on disk / URL
    → parse_document()          extract text
    → RAGPipeline.ingest_text() chunk → embed → store
    → update DB status          PENDING → READY or FAILED

Called as a background task from the upload endpoint.
Also callable directly for URL ingestion.

Usage:
    pipeline = DocumentIngestionPipeline()

    # Ingest an uploaded file
    result = await pipeline.ingest_file(
        file_path=Path("/uploads/report.pdf"),
        doc_id="uuid-...",
        filename="report.pdf",
        user_id="user-123",
    )

    # Ingest a URL
    result = await pipeline.ingest_url(
        url="https://example.com/article",
        doc_id="uuid-...",
        user_id="user-123",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from backend.rag.pipeline import RAGPipeline, IngestionResult, get_rag_pipeline
from backend.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentIngestionResult:
    """Result from the full document ingestion pipeline."""
    doc_id: str
    filename: str
    success: bool
    chunks_stored: int = 0
    text_length: int = 0
    error: Optional[str] = None


class DocumentIngestionPipeline:
    """
    Full pipeline: parse file → extract text → chunk → embed → store.
    Wraps RAGPipeline with file parsing and DB status updates.
    """

    def __init__(self, rag_pipeline: Optional[RAGPipeline] = None):
        self._rag = rag_pipeline or get_rag_pipeline()

    async def ingest_file(
        self,
        file_path: Path,
        doc_id: str,
        filename: str,
        user_id: str,
        doc_type: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> DocumentIngestionResult:
        """
        Parse a file from disk and ingest it into the knowledge base.
        Updates the Document status in PostgreSQL on completion.
        """
        logger.info(
            "file_ingestion_started",
            doc_id=doc_id,
            filename=filename,
            path=str(file_path),
        )

        # 1. Parse
        try:
            from backend.ingestion.parsers import parse_document
            text = await parse_document(file_path)
        except Exception as e:
            logger.error("file_parse_failed", doc_id=doc_id, error=str(e))
            await self._update_db_status(doc_id, "failed", error=str(e))
            return DocumentIngestionResult(
                doc_id=doc_id, filename=filename,
                success=False, error=f"Parse failed: {e}"
            )

        if not text or not text.strip():
            logger.warning("file_empty_after_parsing", doc_id=doc_id)
            await self._update_db_status(doc_id, "failed", error="No text extracted")
            return DocumentIngestionResult(
                doc_id=doc_id, filename=filename,
                success=False, error="No text could be extracted from file"
            )

        # 2. Ingest into RAG
        result = await self._ingest_text(
            text=text,
            doc_id=doc_id,
            filename=filename,
            user_id=user_id,
            doc_type=doc_type or Path(filename).suffix.lstrip(".").lower(),
            metadata=metadata,
        )

        return DocumentIngestionResult(
            doc_id=doc_id,
            filename=filename,
            success=result.success,
            chunks_stored=result.chunks_stored,
            text_length=len(text),
            error=result.error,
        )

    async def ingest_url(
        self,
        url: str,
        doc_id: str,
        user_id: str,
        metadata: Optional[dict] = None,
    ) -> DocumentIngestionResult:
        """Fetch a URL and ingest its content."""
        logger.info("url_ingestion_started", doc_id=doc_id, url=url)

        try:
            from backend.ingestion.parsers.url_parser import parse_url
            text = await parse_url(url)
        except Exception as e:
            logger.error("url_parse_failed", doc_id=doc_id, url=url, error=str(e))
            await self._update_db_status(doc_id, "failed", error=str(e))
            return DocumentIngestionResult(
                doc_id=doc_id, filename=url,
                success=False, error=f"URL fetch failed: {e}"
            )

        result = await self._ingest_text(
            text=text,
            doc_id=doc_id,
            filename=url,
            user_id=user_id,
            doc_type="url",
            metadata={**(metadata or {}), "source_url": url},
        )

        return DocumentIngestionResult(
            doc_id=doc_id,
            filename=url,
            success=result.success,
            chunks_stored=result.chunks_stored,
            text_length=len(text),
            error=result.error,
        )

    async def _ingest_text(
        self,
        text: str,
        doc_id: str,
        filename: str,
        user_id: str,
        doc_type: str,
        metadata: Optional[dict],
    ) -> IngestionResult:
        """Run RAG ingestion and update DB status."""
        result = await self._rag.ingest_text(
            text=text,
            doc_id=doc_id,
            filename=filename,
            doc_type=doc_type,
            metadata={**(metadata or {}), "user_id": user_id},
        )

        status = "ready" if result.success else "failed"
        try:
            await self._update_db_status(
                doc_id, status,
                chunk_count=result.chunks_stored,
                error=result.error,
            )
        except Exception as e:
            # DB update failure must not abort the ingestion result
            from backend.observability.logger import get_logger
            get_logger(__name__).warning("db_status_update_failed", doc_id=doc_id, error=str(e))

        return result

    async def _update_db_status(
        self,
        doc_id: str,
        status: str,
        chunk_count: int = 0,
        error: Optional[str] = None,
    ) -> None:
        """Update Document record status in PostgreSQL."""
        try:
            from backend.db.session import get_db_context
            from backend.db.repositories.base_repo import BaseRepository
            from backend.db.models.document import Document

            class DocRepo(BaseRepository[Document]):
                model = Document

            async with get_db_context() as db:
                import uuid
                await DocRepo().update(
                    db,
                    uuid.UUID(doc_id),
                    status=status,
                    chunk_count=chunk_count,
                    error_message=error,
                )
        except Exception as e:
            # DB update failure should not abort ingestion result
            logger.warning("db_status_update_failed", doc_id=doc_id, error=str(e))


# Module-level singleton
_pipeline: Optional[DocumentIngestionPipeline] = None


def get_ingestion_pipeline() -> DocumentIngestionPipeline:
    """Get the global DocumentIngestionPipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = DocumentIngestionPipeline()
    return _pipeline