"""
Search Endpoint
===============
POST /search — Semantic search across the knowledge base.
Uses get_retriever() so the embedder is mockable via dependency injection.
"""

import time
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.api.dependencies import CurrentUser, DBSession
from backend.vectordb.base import CollectionNotFoundError
from backend.config.settings import get_settings
from backend.observability.logger import get_logger
from backend.rag.retriever import get_retriever

logger = get_logger(__name__)
router = APIRouter(prefix="/search", tags=["Search"])


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    collection: Optional[str] = None
    score_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    model_config = {"json_schema_extra": {
        "examples": [{"query": "What is transformer architecture?", "top_k": 5}]
    }}


class SearchResult(BaseModel):
    document_id: str
    filename: str
    content: str
    score: float
    chunk_index: int
    metadata: dict = {}


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total_found: int
    latency_ms: float


@router.post("", response_model=SearchResponse)
async def search_knowledge_base(
    request: SearchRequest,
    user: CurrentUser,
    db: DBSession,
) -> SearchResponse:
    """
    Semantic search across the user's knowledge base.
    Embeds the query then retrieves the most similar document chunks.
    """
    settings = get_settings()
    start = time.perf_counter()
    collection = request.collection or settings.qdrant_collection_documents

    logger.info(
        "search_request",
        query=request.query[:100],
        top_k=request.top_k,
        collection=collection,
        user_id=str(user.id),
    )

    try:
        retriever = get_retriever()
        retrieved = await retriever.retrieve(
            query=request.query,
            collection=collection,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
        )
    except Exception as e:
        logger.error("search_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Search error: {str(e)}",
        )

    latency_ms = (time.perf_counter() - start) * 1000

    results = [
        SearchResult(
            document_id=doc.doc_id or doc.source,
            filename=doc.source,
            content=doc.content,
            score=round(doc.score, 4),
            chunk_index=doc.chunk_index,
        )
        for doc in retrieved
    ]

    return SearchResponse(
        query=request.query,
        results=results,
        total_found=len(results),
        latency_ms=round(latency_ms, 2),
    )