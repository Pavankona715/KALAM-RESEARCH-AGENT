"""
Search Endpoint
===============
POST /search — Semantic search across the knowledge base.

Returns ranked document chunks relevant to the query.
Used for RAG retrieval and direct knowledge base exploration.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.api.dependencies import CurrentUser, DBSession
from backend.observability.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/search", tags=["Search"])


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    collection: Optional[str] = None   # Search specific collection or all
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    model_config = {"json_schema_extra": {
        "examples": [{
            "query": "What is transformer architecture?",
            "top_k": 5,
        }]
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

    Returns the most relevant document chunks for the given query,
    ranked by semantic similarity score.

    Note: Requires Qdrant (Step 7) and embeddings (Step 8) to be configured.
    Currently returns stub response.
    """
    import time
    start = time.perf_counter()

    logger.info(
        "search_request",
        query=request.query[:100],
        top_k=request.top_k,
        user_id=str(user.id),
    )

    # TODO: Replace stub with real vector search (Step 7 & 8)
    stub_results: list[SearchResult] = []
    latency_ms = (time.perf_counter() - start) * 1000

    return SearchResponse(
        query=request.query,
        results=stub_results,
        total_found=0,
        latency_ms=round(latency_ms, 2),
    )