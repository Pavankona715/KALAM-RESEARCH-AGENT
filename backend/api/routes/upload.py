"""
Upload Endpoint
===============
POST /upload — Upload a document for ingestion into the knowledge base.
GET  /upload/documents — List user's uploaded documents.
DELETE /upload/documents/{doc_id} — Delete a document.

The actual parsing/chunking/embedding happens asynchronously after upload.
"""

import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from backend.api.dependencies import CurrentUser, DBSession, Pagination
from backend.config.settings import get_settings
from backend.db.models.document import Document, DocumentStatus, DocumentType
from backend.db.repositories.base_repo import BaseRepository
from backend.observability.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/upload", tags=["Documents"])


# Simple document repository inline for now (moved to proper file in Step 11)
class DocumentRepository(BaseRepository[Document]):
    model = Document

    async def get_for_user(self, db, user_id, *, skip=0, limit=20):
        from sqlalchemy import select
        result = await db.execute(
            select(Document)
            .where(Document.owner_id == user_id)
            .order_by(Document.created_at.desc())
            .offset(skip).limit(limit)
        )
        return list(result.scalars().all())


doc_repo = DocumentRepository()


class DocumentResponse(BaseModel):
    id: uuid.UUID
    filename: str
    doc_type: str
    status: str
    file_size_bytes: Optional[int]
    chunk_count: int
    created_at: str


def _ext_to_doc_type(extension: str) -> str:
    mapping = {"pdf": "pdf", "docx": "docx", "txt": "txt", "md": "md", "html": "html"}
    return mapping.get(extension.lower(), "txt")


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    db: DBSession,
    user: CurrentUser,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
) -> DocumentResponse:
    """
    Upload a document for ingestion.

    Returns 202 Accepted immediately. Ingestion runs asynchronously.
    Poll GET /upload/documents/{id} to check processing status.

    Supported formats: PDF, DOCX, TXT, MD, HTML
    Max size: configurable via MAX_UPLOAD_SIZE_MB env var
    """
    settings = get_settings()

    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Filename is required",
        )

    extension = Path(file.filename).suffix.lstrip(".").lower()
    if extension not in settings.allowed_extensions_list:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"File type '.{extension}' not supported. "
                   f"Allowed: {settings.allowed_extensions}",
        )

    # Read and size-check
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size {size_mb:.1f}MB exceeds limit of {settings.max_upload_size_mb}MB",
        )

    # Save to upload directory
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = f"{uuid.uuid4()}.{extension}"
    file_path = upload_dir / safe_filename

    with open(file_path, "wb") as f:
        f.write(content)

    # Create DB record
    doc = await doc_repo.create(
        db,
        owner_id=user.id,
        filename=safe_filename,
        original_filename=file.filename,
        doc_type=_ext_to_doc_type(extension),
        file_size_bytes=len(content),
        file_path=str(file_path),
        title=title or Path(file.filename).stem,
        description=description,
        status=DocumentStatus.PENDING.value,
        content_preview="Processing...",
    )

    logger.info(
        "document_uploaded",
        doc_id=str(doc.id),
        filename=file.filename,
        size_mb=round(size_mb, 2),
        user_id=str(user.id),
    )

    # TODO: Trigger async ingestion task (Step 11)
    # await ingestion_pipeline.ingest_async(doc.id)

    return DocumentResponse(
        id=doc.id,
        filename=doc.original_filename,
        doc_type=doc.doc_type,
        status=doc.status,
        file_size_bytes=doc.file_size_bytes,
        chunk_count=doc.chunk_count,
        created_at=doc.created_at.isoformat(),
    )


@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents(
    db: DBSession,
    user: CurrentUser,
    pagination: Pagination,
) -> list[DocumentResponse]:
    """List all documents uploaded by the current user."""
    docs = await doc_repo.get_for_user(
        db, user.id, skip=pagination.skip, limit=pagination.limit
    )
    return [
        DocumentResponse(
            id=d.id,
            filename=d.original_filename,
            doc_type=d.doc_type,
            status=d.status,
            file_size_bytes=d.file_size_bytes,
            chunk_count=d.chunk_count,
            created_at=d.created_at.isoformat(),
        )
        for d in docs
    ]


@router.get("/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: uuid.UUID,
    db: DBSession,
    user: CurrentUser,
) -> DocumentResponse:
    """Get status and metadata of a specific document."""
    doc = await doc_repo.get_by_id(db, doc_id)

    if not doc or doc.owner_id != user.id:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    return DocumentResponse(
        id=doc.id,
        filename=doc.original_filename,
        doc_type=doc.doc_type,
        status=doc.status,
        file_size_bytes=doc.file_size_bytes,
        chunk_count=doc.chunk_count,
        created_at=doc.created_at.isoformat(),
    )


@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    doc_id: uuid.UUID,
    db: DBSession,
    user: CurrentUser,
) -> None:
    """Delete a document and its vectors from the knowledge base."""
    doc = await doc_repo.get_by_id(db, doc_id)

    if not doc or doc.owner_id != user.id:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    # TODO: Remove from Qdrant (Step 7)
    # await vector_db.delete_document(doc_id)

    # Delete file from disk
    if doc.file_path:
        Path(doc.file_path).unlink(missing_ok=True)

    await doc_repo.delete(db, doc_id)
    logger.info("document_deleted", doc_id=str(doc_id), user_id=str(user.id))