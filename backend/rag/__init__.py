"""
RAG Package
===========
Retrieval Augmented Generation pipeline.

Import from here:
    from backend.rag import RAGPipeline, Retriever, get_rag_pipeline, get_retriever
"""

from backend.rag.chunker import RecursiveChunker, FixedSizeChunker, TextChunk, get_chunker
from backend.rag.embedder import Embedder, get_embedder
from backend.rag.retriever import Retriever, get_retriever
from backend.rag.pipeline import RAGPipeline, IngestionResult, get_rag_pipeline

__all__ = [
    "RecursiveChunker", "FixedSizeChunker", "TextChunk", "get_chunker",
    "Embedder", "get_embedder",
    "Retriever", "get_retriever",
    "RAGPipeline", "IngestionResult", "get_rag_pipeline",
]