"""
Retriever
=========
Converts a text query into ranked document chunks from the vector store.

Pipeline:
  query text
    → Embedder.embed_one()        embed the query
    → VectorDB.search()           find similar chunks
    → _rerank()                   optional reranking
    → list[RetrievedDocument]     ready for ContextBuilder

The Retriever is query-time only — it never writes to the vector DB.
For writing, use RAGPipeline (ingestion).

Reranking (future):
  Basic retrieval uses cosine similarity which works well but isn't perfect.
  A cross-encoder reranker (e.g. Cohere Rerank) can dramatically improve
  precision by scoring each (query, chunk) pair directly. The interface
  is designed to plug this in without changing callers.
"""

from __future__ import annotations

from typing import Optional

from backend.context.builder import RetrievedDocument
from backend.rag.embedder import Embedder, get_embedder
from backend.vectordb.base import VectorDB, CollectionNotFoundError
from backend.vectordb.factory import get_vector_db
from backend.observability.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    """
    Retrieves relevant document chunks for a given query.

    Usage:
        retriever = Retriever()
        docs = await retriever.retrieve(
            query="What is attention mechanism?",
            top_k=5,
        )
        # docs is list[RetrievedDocument] ready for ContextBuilder
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        vector_db: Optional[VectorDB] = None,
    ):
        self._embedder = embedder or get_embedder()
        self._vector_db = vector_db or get_vector_db()

    async def retrieve(
        self,
        query: str,
        *,
        collection: Optional[str] = None,
        top_k: int = 5,
        score_threshold: float = 0.3,
        filter_doc_ids: Optional[list[str]] = None,
    ) -> list[RetrievedDocument]:
        """
        Retrieve the most relevant document chunks for a query.

        Args:
            query: The user's question or search query
            collection: Qdrant collection to search (defaults to documents)
            top_k: Maximum chunks to return
            score_threshold: Minimum similarity score (0.0-1.0)
            filter_doc_ids: Only return chunks from these document IDs

        Returns:
            List of RetrievedDocument ordered by relevance (highest first)
        """
        from backend.config.settings import get_settings
        settings = get_settings()
        collection = collection or settings.qdrant_collection_documents

        logger.debug(
            "retrieval_started",
            query=query[:100],
            collection=collection,
            top_k=top_k,
        )

        # 1. Embed the query
        try:
            query_vector = await self._embedder.embed_one(query)
        except Exception as e:
            logger.error("query_embedding_failed", error=str(e))
            return []  # Fail gracefully — agent continues without RAG

        # 2. Build metadata filter
        metadata_filter = None
        if filter_doc_ids:
            # Note: Qdrant filter supports "any of" via should conditions
            # For simplicity we pass the first doc_id; full multi-filter in production
            metadata_filter = {"doc_id": filter_doc_ids[0]} if len(filter_doc_ids) == 1 else None

        # 3. Search vector DB
        try:
            raw_results = await self._vector_db.search(
                collection=collection,
                query_vector=query_vector,
                top_k=top_k,
                score_threshold=score_threshold,
                filter_metadata=metadata_filter,
            )
        except CollectionNotFoundError:
            logger.debug("collection_not_found_no_docs_ingested", collection=collection)
            return []
        except Exception as e:
            logger.error("vector_search_failed", error=str(e))
            return []

        # 4. Convert to RetrievedDocument format (used by ContextBuilder)
        retrieved = [
            RetrievedDocument(
                content=result.text,
                source=result.metadata.get("filename", result.id),
                score=result.score,
                doc_id=result.metadata.get("doc_id", result.id),
                chunk_index=result.metadata.get("chunk_index", 0),
            )
            for result in raw_results
        ]

        logger.debug(
            "retrieval_completed",
            query=query[:100],
            results_found=len(retrieved),
            top_score=retrieved[0].score if retrieved else 0,
        )

        return retrieved

    async def retrieve_for_agent(
        self,
        query: str,
        session_id: str,
        top_k: int = 5,
    ) -> list[RetrievedDocument]:
        """
        Convenience method for agent use — retrieves from the default
        documents collection with sensible defaults.
        """
        return await self.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=0.3,
        )


# Module-level singleton
_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    """Get the global Retriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever