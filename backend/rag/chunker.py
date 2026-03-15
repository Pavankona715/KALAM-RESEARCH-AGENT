"""
Document Chunker
================
Splits documents into overlapping chunks suitable for embedding.

Two strategies:
1. RecursiveChunker  — splits on paragraph → sentence → word boundaries
                       Best for prose documents (PDFs, articles)
2. FixedSizeChunker  — splits at exactly N characters with overlap
                       Best for structured data, code, logs

Overlap is critical: if a concept spans a chunk boundary, overlap
ensures it appears fully in at least one chunk, so retrieval finds it.

Chunk size guidance:
  - 512 tokens (~2000 chars): good for factual retrieval
  - 1024 tokens (~4000 chars): good for reasoning over longer passages
  - Too large: LLM context fills up fast, less precise retrieval
  - Too small: concepts split across chunks, context lost
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class TextChunk:
    """A chunk of text with its position in the source document."""
    text: str
    chunk_index: int
    start_char: int       # Character offset in original document
    end_char: int
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


class BaseChunker(ABC):
    """Abstract base for all chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[TextChunk]:
        """Split text into chunks. Returns list of TextChunk objects."""
        ...


class RecursiveChunker(BaseChunker):
    """
    Recursively splits text using a hierarchy of separators.

    Tries to split on natural boundaries in order:
    1. Double newlines (paragraphs)
    2. Single newlines
    3. Sentences (. ! ?)
    4. Spaces (words)
    5. Characters (last resort)

    Falls back to the next separator only when a segment is still
    too large after splitting on the current one.

    This preserves semantic coherence much better than fixed-size splitting.
    """

    # Separator hierarchy: try these in order
    SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 1500,      # Target chars per chunk
        chunk_overlap: int = 200,    # Chars to repeat between chunks
        min_chunk_size: int = 100,   # Drop chunks smaller than this
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[TextChunk]:
        """Split text into overlapping chunks."""
        if not text or not text.strip():
            return []

        # Clean up excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        raw_chunks = self._split_recursive(text, self.SEPARATORS)
        return self._merge_with_overlap(raw_chunks, text, metadata or {})

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split using separator hierarchy."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        separator = separators[0] if separators else ""
        remaining_seps = separators[1:] if len(separators) > 1 else []

        if separator:
            splits = text.split(separator)
        else:
            # Character-level last resort
            splits = list(text)

        chunks = []
        current = ""

        for split in splits:
            candidate = current + (separator if current else "") + split

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If this single split is too large, recurse
                if len(split) > self.chunk_size and remaining_seps:
                    chunks.extend(self._split_recursive(split, remaining_seps))
                    current = ""
                else:
                    current = split

        if current:
            chunks.append(current)

        return [c for c in chunks if c.strip()]

    def _merge_with_overlap(
        self,
        raw_chunks: list[str],
        original_text: str,
        metadata: dict,
    ) -> list[TextChunk]:
        """Convert raw string chunks to TextChunk objects with overlap."""
        result = []
        search_start = 0

        for i, chunk_text in enumerate(raw_chunks):
            if len(chunk_text) < self.min_chunk_size:
                continue

            # Find position in original text
            start = original_text.find(chunk_text, search_start)
            if start == -1:
                start = search_start
            end = start + len(chunk_text)
            search_start = max(0, end - self.chunk_overlap)

            result.append(TextChunk(
                text=chunk_text.strip(),
                chunk_index=len(result),
                start_char=start,
                end_char=end,
                metadata=dict(metadata),
            ))

        return result


class FixedSizeChunker(BaseChunker):
    """
    Splits text at fixed character intervals with overlap.
    Simple and predictable — good for structured content.
    """

    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[TextChunk]:
        if not text or not text.strip():
            return []

        chunks = []
        start = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(TextChunk(
                    text=chunk_text,
                    chunk_index=len(chunks),
                    start_char=start,
                    end_char=end,
                    metadata=dict(metadata or {}),
                ))

            if end >= len(text):
                break
            start += step

        return chunks


def get_chunker(strategy: str = "recursive", **kwargs) -> BaseChunker:
    """
    Factory function for chunkers.

    Args:
        strategy: "recursive" or "fixed"
        **kwargs: Passed to chunker constructor (chunk_size, chunk_overlap, etc.)
    """
    if strategy == "recursive":
        return RecursiveChunker(**kwargs)
    elif strategy == "fixed":
        return FixedSizeChunker(**kwargs)
    else:
        raise ValueError(f"Unknown chunking strategy: '{strategy}'. Use 'recursive' or 'fixed'.")