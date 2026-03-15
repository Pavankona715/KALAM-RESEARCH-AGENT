"""
Embedder
========
Converts text chunks into dense vector embeddings.

Responsibilities:
- Batch texts for efficient API usage (one call per N texts)
- Retry on transient failures
- Cache embeddings to avoid re-embedding the same text twice
- Track token usage for cost monitoring

The embedder is stateless — it doesn't know about documents or chunks.
It only knows: text in → vector out.

Supported models (via LiteLLM):
  - text-embedding-3-small  (OpenAI, default, 1536 dims, cheap)
  - text-embedding-3-large  (OpenAI, 3072 dims, better quality)
  - text-embedding-ada-002  (OpenAI, legacy)
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Optional

from backend.observability.logger import get_logger

logger = get_logger(__name__)

# Maximum texts per API call — OpenAI supports up to 2048 but
# smaller batches give better error isolation
DEFAULT_BATCH_SIZE = 100


class Embedder:
    """
    Generates embeddings for text using the configured model.

    Usage:
        embedder = Embedder()

        # Single text
        vector = await embedder.embed_one("What is RAG?")

        # Batch (more efficient)
        vectors = await embedder.embed_many(["text1", "text2", "text3"])
    """

    def __init__(
        self,
        model: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        use_cache: bool = True,
    ):
        from backend.config.settings import get_settings
        settings = get_settings()

        self._model = model or settings.embedding_model
        self._batch_size = batch_size
        self._use_cache = use_cache
        self._cache: dict[str, list[float]] = {}  # text hash → vector
        self._total_texts_embedded = 0

    async def embed_one(self, text: str) -> list[float]:
        """Embed a single text string."""
        results = await self.embed_many([text])
        return results[0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts efficiently using batching.

        Returns vectors in the same order as input texts.
        Caches results to avoid re-embedding identical texts.
        """
        if not texts:
            return []

        # Separate cached from uncached
        results: list[Optional[list[float]]] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        if self._use_cache:
            for i, text in enumerate(texts):
                cache_key = self._hash(text)
                if cache_key in self._cache:
                    results[i] = self._cache[cache_key]
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts

        # Embed uncached texts in batches
        if uncached_texts:
            embeddings = await self._embed_in_batches(uncached_texts)

            for idx, embedding in zip(uncached_indices, embeddings):
                results[idx] = embedding
                if self._use_cache:
                    self._cache[self._hash(texts[idx])] = embedding

            self._total_texts_embedded += len(uncached_texts)

        return [r for r in results if r is not None]

    async def _embed_in_batches(
        self, texts: list[str]
    ) -> list[list[float]]:
        """Split texts into batches and embed each batch."""
        all_embeddings = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]

            logger.debug(
                "embedding_batch",
                model=self._model,
                batch_size=len(batch),
                batch_num=i // self._batch_size + 1,
            )

            embeddings = await self._call_api_with_retry(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    async def _call_api_with_retry(
        self,
        texts: list[str],
        max_retries: int = 3,
    ) -> list[list[float]]:
        """Call the embedding API with exponential backoff retry."""
        import litellm

        last_error = None
        for attempt in range(max_retries):
            try:
                response = await litellm.aembedding(
                    model=self._model,
                    input=texts,
                )
                return [item["embedding"] for item in response.data]

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(
                        "embedding_retry",
                        attempt=attempt + 1,
                        wait_seconds=wait,
                        error=str(e)[:100],
                    )
                    await asyncio.sleep(wait)

        raise RuntimeError(
            f"Embedding failed after {max_retries} attempts: {last_error}"
        )

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    @property
    def total_embedded(self) -> int:
        return self._total_texts_embedded

    @staticmethod
    def _hash(text: str) -> str:
        """Create a stable cache key from text content."""
        return hashlib.md5(text.encode()).hexdigest()


# Module-level singleton
_embedder: Optional[Embedder] = None


def get_embedder() -> Embedder:
    """Get the global Embedder singleton."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder