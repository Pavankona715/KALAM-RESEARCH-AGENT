"""
RAG Ingestion Pipeline
======================
Orchestrates the full document ingestion flow:

  raw text
    → chunk (RecursiveChunker)
    → embed (Embedder)
    → store (VectorDB.upsert)
    → update document status in PostgreSQL

This module is called by:
  - POST /upload (Step 2) after a file is saved to disk
  - URL ingestion (Step 11)

The pipeline is intentionally simple and synchronous per document.
For production scale, wrap each call in a Celery/ARQ task queue
so ingestion doesn't block HTTP responses.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional

from backend.rag.chunker import RecursiveChunker, TextChunk
from backend.rag.embedder import Embedder, get_embedder
from backend.vectordb.base import DocumentChunk, VectorDB
from backend.vectordb.factory import get_vector_db
from backend.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class IngestionResult:
    """Result from ingesting a single document."""
    doc_id: str
    filename: str
    chunks_created: int
    chunks_embedded: int
    chunks_stored: int
    success: bool
    error: Optional[str] = None


class RAGPipeline:
    """
    End-to-end document ingestion pipeline.

    Usage:
        pipeline = RAGPipeline()

        result = await pipeline.ingest_text(
            text="Full document content...",
            doc_id="doc-uuid-123",
            filename="research_paper.pdf",
            metadata={"source": "upload", "user_id": "user-456"},
        )

        print(f"Stored {result.chunks_stored} chunks")
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        vector_db: Optional[VectorDB] = None,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
    ):
        self._embedder = embedder or get_embedder()
        self._vector_db = vector_db or get_vector_db()
        self._chunker = RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    async def ingest_text(
        self,
        text: str,
        doc_id: str,
        filename: str,
        collection: Optional[str] = None,
        metadata: Optional[dict] = None,
        doc_type: str = "txt",
    ) -> IngestionResult:
        """
        Ingest a raw text string into the vector store.

        Args:
            text: The full document text content
            doc_id: Unique document identifier (from PostgreSQL)
            filename: Original filename for display purposes
            collection: Qdrant collection (defaults to settings value)
            metadata: Extra metadata to store with each chunk
            doc_type: File type for metadata (pdf, docx, txt, url)

        Returns:
            IngestionResult with stats about what was stored
        """
        from backend.config.settings import get_settings
        settings = get_settings()
        collection = collection or settings.qdrant_collection_documents

        logger.info(
            "ingestion_started",
            doc_id=doc_id,
            filename=filename,
            text_length=len(text),
            collection=collection,
        )

        try:
            # 1. Chunk
            base_metadata = {
                "doc_id": doc_id,
                "filename": filename,
                "doc_type": doc_type,
                **(metadata or {}),
            }
            text_chunks = self._chunker.chunk(text, metadata=base_metadata)

            if not text_chunks:
                return IngestionResult(
                    doc_id=doc_id,
                    filename=filename,
                    chunks_created=0,
                    chunks_embedded=0,
                    chunks_stored=0,
                    success=True,
                )

            logger.debug(
                "document_chunked",
                doc_id=doc_id,
                chunk_count=len(text_chunks),
                avg_chunk_size=sum(c.char_count for c in text_chunks) // len(text_chunks),
            )

            # 2. Embed all chunks in one batch call
            texts_to_embed = [chunk.text for chunk in text_chunks]
            embeddings = await self._embedder.embed_many(texts_to_embed)

            if len(embeddings) != len(text_chunks):
                raise RuntimeError(
                    f"Embedding count mismatch: {len(embeddings)} embeddings "
                    f"for {len(text_chunks)} chunks"
                )

            # 3. Build DocumentChunk objects
            doc_chunks = [
                DocumentChunk(
                    id=f"{doc_id}_chunk_{chunk.chunk_index}",
                    text=chunk.text,
                    embedding=embedding,
                    metadata={
                        **chunk.metadata,
                        "chunk_index": chunk.chunk_index,
                        "char_count": chunk.char_count,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                    },
                )
                for chunk, embedding in zip(text_chunks, embeddings)
            ]

            # 4. Store in vector DB
            stored_count = await self._vector_db.upsert(collection, doc_chunks)

            logger.info(
                "ingestion_completed",
                doc_id=doc_id,
                filename=filename,
                chunks_stored=stored_count,
            )

            return IngestionResult(
                doc_id=doc_id,
                filename=filename,
                chunks_created=len(text_chunks),
                chunks_embedded=len(embeddings),
                chunks_stored=stored_count,
                success=True,
            )

        except Exception as e:
            logger.error(
                "ingestion_failed",
                doc_id=doc_id,
                filename=filename,
                error=str(e),
            )
            return IngestionResult(
                doc_id=doc_id,
                filename=filename,
                chunks_created=0,
                chunks_embedded=0,
                chunks_stored=0,
                success=False,
                error=str(e),
            )

    async def delete_document(
        self,
        doc_id: str,
        collection: Optional[str] = None,
    ) -> int:
        """
        Remove all chunks for a document from the vector store.
        Called when a user deletes a document.
        """
        from backend.config.settings import get_settings
        settings = get_settings()
        collection = collection or settings.qdrant_collection_documents

        deleted = await self._vector_db.delete_by_doc_id(collection, doc_id)
        logger.info("document_deleted_from_vectordb", doc_id=doc_id, chunks_removed=deleted)
        return deleted


# Module-level singleton
_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Get the global RAGPipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline