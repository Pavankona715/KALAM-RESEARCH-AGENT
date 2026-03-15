"""
RAG Pipeline Unit Tests
=======================
Tests for chunker, embedder, retriever, and full ingestion pipeline.

- Chunker tests: pure Python, no dependencies
- Embedder tests: mock LiteLLM API calls
- Pipeline tests: in-memory Qdrant + mock embedder
- Retriever tests: in-memory Qdrant + mock embedder
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.vectordb.base import (
    DocumentChunk, SearchResult, CollectionInfo,
    VectorDB, CollectionNotFoundError,
)


# ─── VectorDB Data Type Tests ─────────────────────────────────────────────────

class TestDocumentChunk:
    def test_chunk_creation(self):
        chunk = DocumentChunk(
            id="doc1_chunk_0", text="Hello world",
            embedding=[0.1, 0.2, 0.3],
            metadata={"doc_id": "doc1", "filename": "hello.txt"},
        )
        assert chunk.id == "doc1_chunk_0"
        assert len(chunk.embedding) == 3

    def test_default_metadata(self):
        chunk = DocumentChunk(id="c1", text="text", embedding=[0.1])
        assert chunk.metadata == {}


class TestSearchResult:
    def test_doc_id_property(self):
        r = SearchResult(id="c1", text="text", score=0.9, metadata={"doc_id": "my-doc"})
        assert r.doc_id == "my-doc"

    def test_missing_metadata_returns_none(self):
        r = SearchResult(id="c1", text="text", score=0.9)
        assert r.doc_id is None
        assert r.filename is None


# ─── Chunker Tests ────────────────────────────────────────────────────────────

class TestRecursiveChunker:
    def setup_method(self):
        from backend.rag.chunker import RecursiveChunker
        self.chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20, min_chunk_size=10)

    def test_empty_text_returns_empty(self):
        assert self.chunker.chunk("") == []
        assert self.chunker.chunk("   ") == []

    def test_short_text_returns_single_chunk(self):
        text = "This is a short text."
        chunks = self.chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0].text == text.strip()
        assert chunks[0].chunk_index == 0

    def test_long_text_splits_into_multiple_chunks(self):
        text = "paragraph one\n\n" * 20
        chunks = self.chunker.chunk(text)
        assert len(chunks) > 1

    def test_chunks_have_correct_indices(self):
        text = "word " * 200
        chunks = self.chunker.chunk(text)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_metadata_propagated_to_all_chunks(self):
        text = "word " * 200
        meta = {"doc_id": "doc-1", "filename": "test.txt"}
        chunks = self.chunker.chunk(text, metadata=meta)
        for chunk in chunks:
            assert chunk.metadata["doc_id"] == "doc-1"
            assert chunk.metadata["filename"] == "test.txt"

    def test_char_count_property(self):
        from backend.rag.chunker import TextChunk
        chunk = TextChunk(text="hello world", chunk_index=0, start_char=0, end_char=11)
        assert chunk.char_count == 11

    def test_word_count_property(self):
        from backend.rag.chunker import TextChunk
        chunk = TextChunk(text="hello world foo", chunk_index=0, start_char=0, end_char=15)
        assert chunk.word_count == 3

    def test_paragraph_split_respects_boundaries(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunker_small = __import__('backend.rag.chunker', fromlist=['RecursiveChunker']).RecursiveChunker(
            chunk_size=30, chunk_overlap=5, min_chunk_size=5
        )
        chunks = chunker_small.chunk(text)
        assert len(chunks) >= 2


class TestFixedSizeChunker:
    def setup_method(self):
        from backend.rag.chunker import FixedSizeChunker
        self.chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)

    def test_produces_fixed_size_chunks(self):
        text = "x" * 500
        chunks = self.chunker.chunk(text)
        assert len(chunks) > 1
        for chunk in chunks[:-1]:
            assert chunk.char_count <= 100

    def test_overlap_creates_shared_content(self):
        text = "a" * 200
        chunks = self.chunker.chunk(text)
        if len(chunks) >= 2:
            # The end of chunk 0 and start of chunk 1 should overlap
            end_of_first = chunks[0].end_char
            start_of_second = chunks[1].start_char
            assert start_of_second < end_of_first  # Overlap exists

    def test_empty_text(self):
        assert self.chunker.chunk("") == []


class TestGetChunker:
    def test_recursive_strategy(self):
        from backend.rag.chunker import get_chunker, RecursiveChunker
        chunker = get_chunker("recursive")
        assert isinstance(chunker, RecursiveChunker)

    def test_fixed_strategy(self):
        from backend.rag.chunker import get_chunker, FixedSizeChunker
        chunker = get_chunker("fixed")
        assert isinstance(chunker, FixedSizeChunker)

    def test_unknown_strategy_raises(self):
        from backend.rag.chunker import get_chunker
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            get_chunker("semantic")

    def test_kwargs_passed_to_chunker(self):
        from backend.rag.chunker import get_chunker, RecursiveChunker
        chunker = get_chunker("recursive", chunk_size=500, chunk_overlap=50)
        assert isinstance(chunker, RecursiveChunker)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50


# ─── Embedder Tests ───────────────────────────────────────────────────────────

class TestEmbedder:
    def setup_method(self):
        from backend.rag.embedder import Embedder
        self.embedder = Embedder(model="text-embedding-3-small", use_cache=True)

    @pytest.mark.asyncio
    async def test_embed_one(self):
        mock_vector = [0.1] * 1536
        with patch("litellm.aembedding", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = MagicMock(
                data=[{"embedding": mock_vector}]
            )
            vector = await self.embedder.embed_one("Hello world")
        assert len(vector) == 1536
        assert vector[0] == 0.1

    @pytest.mark.asyncio
    async def test_embed_many_returns_correct_count(self):
        texts = ["text1", "text2", "text3"]
        mock_vectors = [[0.1] * 8, [0.2] * 8, [0.3] * 8]
        with patch("litellm.aembedding", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = MagicMock(
                data=[{"embedding": v} for v in mock_vectors]
            )
            vectors = await self.embedder.embed_many(texts)
        assert len(vectors) == 3

    @pytest.mark.asyncio
    async def test_embed_empty_list(self):
        vectors = await self.embedder.embed_many([])
        assert vectors == []

    @pytest.mark.asyncio
    async def test_caching_avoids_duplicate_calls(self):
        mock_vector = [0.1] * 8
        with patch("litellm.aembedding", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = MagicMock(data=[{"embedding": mock_vector}])
            await self.embedder.embed_one("cached text")
            await self.embedder.embed_one("cached text")  # Should use cache
            assert mock_embed.call_count == 1  # API called only once

    @pytest.mark.asyncio
    async def test_cache_size_increments(self):
        mock_vector = [0.1] * 8
        with patch("litellm.aembedding", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = MagicMock(data=[{"embedding": mock_vector}])
            assert self.embedder.cache_size == 0
            await self.embedder.embed_one("new text")
            assert self.embedder.cache_size == 1

    def test_hash_is_deterministic(self):
        from backend.rag.embedder import Embedder
        h1 = Embedder._hash("same text")
        h2 = Embedder._hash("same text")
        assert h1 == h2

    def test_hash_differs_for_different_text(self):
        from backend.rag.embedder import Embedder
        h1 = Embedder._hash("text a")
        h2 = Embedder._hash("text b")
        assert h1 != h2


# ─── Qdrant Adapter Tests ─────────────────────────────────────────────────────

def make_chunk(text="Sample", doc_id="doc-1", chunk_index=0, dim=4):
    return DocumentChunk(
        id=f"{doc_id}_chunk_{chunk_index}",
        text=text,
        embedding=[0.1 * (chunk_index + 1)] * dim,
        metadata={"doc_id": doc_id, "filename": f"{doc_id}.txt", "chunk_index": chunk_index},
    )


@pytest.fixture
async def in_memory_db():
    from backend.vectordb.qdrant_adapter import QdrantAdapter
    db = QdrantAdapter(in_memory=True)
    yield db
    await db.close()


@pytest.fixture
async def populated_db(in_memory_db):
    await in_memory_db.create_collection("test_docs", dimension=4)
    chunks = [
        make_chunk("Python is a programming language", "doc-1", 0),
        make_chunk("Python supports async programming", "doc-1", 1),
        make_chunk("JavaScript runs in the browser", "doc-2", 0),
    ]
    await in_memory_db.upsert("test_docs", chunks)
    return in_memory_db


class TestQdrantAdapterCollections:
    @pytest.mark.asyncio
    async def test_create_collection(self, in_memory_db):
        assert await in_memory_db.create_collection("my_col", dimension=4) is True

    @pytest.mark.asyncio
    async def test_create_collection_idempotent(self, in_memory_db):
        await in_memory_db.create_collection("my_col", dimension=4)
        assert await in_memory_db.create_collection("my_col", dimension=4) is False

    @pytest.mark.asyncio
    async def test_collection_exists(self, in_memory_db):
        await in_memory_db.create_collection("exists_col", dimension=4)
        assert await in_memory_db.collection_exists("exists_col") is True
        assert await in_memory_db.collection_exists("nonexistent") is False

    @pytest.mark.asyncio
    async def test_get_collection_info(self, in_memory_db):
        await in_memory_db.create_collection("info_col", dimension=8)
        info = await in_memory_db.get_collection_info("info_col")
        assert info is not None
        assert info.name == "info_col"
        assert info.dimension == 8
        assert info.vector_count == 0
        assert isinstance(info.disk_size_bytes, int)  # Just check it's an int, not the value

    @pytest.mark.asyncio
    async def test_get_collection_info_nonexistent(self, in_memory_db):
        assert await in_memory_db.get_collection_info("ghost") is None


class TestQdrantAdapterUpsert:
    @pytest.mark.asyncio
    async def test_upsert_creates_collection_automatically(self, in_memory_db):
        count = await in_memory_db.upsert("auto_col", [make_chunk()])
        assert count == 1
        assert await in_memory_db.collection_exists("auto_col")

    @pytest.mark.asyncio
    async def test_upsert_returns_count(self, in_memory_db):
        await in_memory_db.create_collection("count_col", dimension=4)
        chunks = [make_chunk(chunk_index=i) for i in range(5)]
        assert await in_memory_db.upsert("count_col", chunks) == 5

    @pytest.mark.asyncio
    async def test_upsert_empty_list(self, in_memory_db):
        assert await in_memory_db.upsert("any_col", []) == 0

    @pytest.mark.asyncio
    async def test_upsert_increases_count(self, in_memory_db):
        await in_memory_db.create_collection("grow_col", dimension=4)
        assert await in_memory_db.count("grow_col") == 0
        await in_memory_db.upsert("grow_col", [make_chunk("first")])
        assert await in_memory_db.count("grow_col") == 1

    @pytest.mark.asyncio
    async def test_upsert_is_idempotent(self, in_memory_db):
        await in_memory_db.create_collection("idem_col", dimension=4)
        chunk = make_chunk("same chunk")
        await in_memory_db.upsert("idem_col", [chunk])
        await in_memory_db.upsert("idem_col", [chunk])
        assert await in_memory_db.count("idem_col") == 1


class TestQdrantAdapterSearch:
    @pytest.mark.asyncio
    async def test_search_returns_results(self, populated_db):
        results = await populated_db.search("test_docs", [0.1, 0.1, 0.1, 0.1], top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_results_have_text(self, populated_db):
        results = await populated_db.search("test_docs", [0.1, 0.1, 0.1, 0.1], top_k=1)
        assert len(results[0].text) > 0

    @pytest.mark.asyncio
    async def test_search_respects_top_k(self, populated_db):
        results = await populated_db.search("test_docs", [0.1, 0.1, 0.1, 0.1], top_k=2)
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_search_results_ordered_by_score(self, populated_db):
        results = await populated_db.search("test_docs", [0.1, 0.1, 0.1, 0.1], top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_search_nonexistent_collection_raises(self, in_memory_db):
        with pytest.raises(CollectionNotFoundError):
            await in_memory_db.search("ghost_col", [0.1, 0.1, 0.1, 0.1])


class TestQdrantAdapterDelete:
    @pytest.mark.asyncio
    async def test_delete_by_doc_id(self, populated_db):
        count_before = await populated_db.count("test_docs")
        deleted = await populated_db.delete_by_doc_id("test_docs", "doc-1")
        assert deleted == 2
        assert await populated_db.count("test_docs") == count_before - 2

    @pytest.mark.asyncio
    async def test_delete_from_nonexistent_collection(self, in_memory_db):
        assert await in_memory_db.delete_by_doc_id("ghost_col", "doc-1") == 0


# ─── VectorDB Protocol + Factory Tests ───────────────────────────────────────

class TestVectorDBProtocol:
    def test_qdrant_adapter_satisfies_protocol(self):
        from backend.vectordb.qdrant_adapter import QdrantAdapter
        assert isinstance(QdrantAdapter(in_memory=True), VectorDB)

    def test_str_to_uuid_is_deterministic(self):
        from backend.vectordb.qdrant_adapter import QdrantAdapter
        assert QdrantAdapter._str_to_uuid("x") == QdrantAdapter._str_to_uuid("x")

    def test_str_to_uuid_unique_per_input(self):
        from backend.vectordb.qdrant_adapter import QdrantAdapter
        assert QdrantAdapter._str_to_uuid("a") != QdrantAdapter._str_to_uuid("b")


class TestVectorDBFactory:
    def test_create_qdrant(self):
        from backend.vectordb.factory import create_vector_db
        assert isinstance(create_vector_db("qdrant", in_memory=True), VectorDB)

    def test_unknown_backend_raises(self):
        from backend.vectordb.factory import create_vector_db
        with pytest.raises(ValueError):
            create_vector_db("pinecone")

    def test_in_memory_returns_new_instance(self):
        from backend.vectordb.factory import get_vector_db
        assert get_vector_db(in_memory=True) is not get_vector_db(in_memory=True)

    @pytest.mark.asyncio
    async def test_ensure_collections_exist(self):
        from backend.vectordb.factory import ensure_collections_exist
        from backend.vectordb.qdrant_adapter import QdrantAdapter
        from backend.config.settings import get_settings
        settings = get_settings()
        db = QdrantAdapter(in_memory=True)
        await ensure_collections_exist(vector_db=db)
        assert await db.collection_exists(settings.qdrant_collection_documents)
        assert await db.collection_exists(settings.qdrant_collection_memory)
        await db.close()


# ─── RAG Pipeline Tests ───────────────────────────────────────────────────────

class TestRAGPipeline:
    def _make_pipeline(self, dim=8):
        from backend.rag.pipeline import RAGPipeline
        from backend.vectordb.qdrant_adapter import QdrantAdapter
        from backend.rag.embedder import Embedder

        mock_embedder = MagicMock(spec=Embedder)
        mock_embedder.embed_many = AsyncMock(
            side_effect=lambda texts: [[0.1] * dim for _ in texts]
        )

        db = QdrantAdapter(in_memory=True)
        return RAGPipeline(embedder=mock_embedder, vector_db=db), db

    @pytest.mark.asyncio
    async def test_ingest_text_success(self):
        pipeline, db = self._make_pipeline()
        # Text must exceed min_chunk_size (100 chars) to produce chunks
        long_text = "This is a test document about Python programming. " * 5
        result = await pipeline.ingest_text(
            text=long_text,
            doc_id="doc-001",
            filename="test.txt",
        )
        assert result.success is True
        assert result.chunks_stored > 0
        assert result.doc_id == "doc-001"

    @pytest.mark.asyncio
    async def test_ingest_empty_text(self):
        pipeline, db = self._make_pipeline()
        result = await pipeline.ingest_text(
            text="", doc_id="doc-002", filename="empty.txt"
        )
        assert result.success is True
        assert result.chunks_stored == 0

    @pytest.mark.asyncio
    async def test_ingest_stores_chunks_in_vectordb(self):
        pipeline, db = self._make_pipeline()
        from backend.config.settings import get_settings
        collection = get_settings().qdrant_collection_documents

        await pipeline.ingest_text(
            text="word " * 100,
            doc_id="doc-003",
            filename="words.txt",
        )
        count = await db.count(collection)
        assert count > 0

    @pytest.mark.asyncio
    async def test_ingest_metadata_passed_through(self):
        pipeline, db = self._make_pipeline()
        result = await pipeline.ingest_text(
            text="document content here " * 10,
            doc_id="doc-004",
            filename="meta.txt",
            metadata={"user_id": "user-123", "source": "upload"},
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_ingest_handles_embedding_failure(self):
        from backend.rag.pipeline import RAGPipeline
        from backend.vectordb.qdrant_adapter import QdrantAdapter
        from backend.rag.embedder import Embedder

        mock_embedder = MagicMock(spec=Embedder)
        mock_embedder.embed_many = AsyncMock(side_effect=RuntimeError("API down"))

        pipeline = RAGPipeline(
            embedder=mock_embedder,
            vector_db=QdrantAdapter(in_memory=True),
        )
        result = await pipeline.ingest_text(
            text="some text " * 10, doc_id="doc-005", filename="fail.txt"
        )
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_delete_document(self):
        pipeline, db = self._make_pipeline()
        from backend.config.settings import get_settings
        collection = get_settings().qdrant_collection_documents

        await pipeline.ingest_text(
            text="word " * 100, doc_id="doc-del", filename="del.txt"
        )
        count_after_ingest = await db.count(collection)
        assert count_after_ingest > 0

        deleted = await pipeline.delete_document("doc-del")
        assert deleted > 0
        assert await db.count(collection) == 0


# ─── Retriever Tests ──────────────────────────────────────────────────────────

class TestRetriever:
    @pytest.fixture
    async def retriever_with_data(self):
        """Async fixture: Retriever backed by in-memory Qdrant with pre-loaded data."""
        from backend.rag.retriever import Retriever
        from backend.vectordb.qdrant_adapter import QdrantAdapter
        from backend.rag.embedder import Embedder
        from backend.config.settings import get_settings

        dim = 4
        mock_embedder = MagicMock(spec=Embedder)
        mock_embedder.embed_one = AsyncMock(return_value=[0.1] * dim)

        db = QdrantAdapter(in_memory=True)
        collection = get_settings().qdrant_collection_documents
        await db.create_collection(collection, dimension=dim)
        chunks = [
            DocumentChunk(
                id=f"doc-1_chunk_{i}",
                text=f"Chunk {i} about Python",
                embedding=[0.1 * (i + 1)] * dim,
                metadata={"doc_id": "doc-1", "filename": "python.txt", "chunk_index": i},
            )
            for i in range(3)
        ]
        await db.upsert(collection, chunks)
        retriever = Retriever(embedder=mock_embedder, vector_db=db)
        yield retriever
        await db.close()

    @pytest.mark.asyncio
    async def test_retrieve_returns_results(self, retriever_with_data):
        results = await retriever_with_data.retrieve("What is Python?", top_k=3)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_retrieve_returns_retrieved_documents(self, retriever_with_data):
        from backend.context.builder import RetrievedDocument
        results = await retriever_with_data.retrieve("Python programming", top_k=2)
        assert all(isinstance(r, RetrievedDocument) for r in results)

    @pytest.mark.asyncio
    async def test_retrieve_empty_collection_returns_empty(self):
        from backend.rag.retriever import Retriever
        from backend.vectordb.qdrant_adapter import QdrantAdapter
        from backend.rag.embedder import Embedder

        mock_embedder = MagicMock(spec=Embedder)
        mock_embedder.embed_one = AsyncMock(return_value=[0.1] * 4)
        db = QdrantAdapter(in_memory=True)

        retriever = Retriever(embedder=mock_embedder, vector_db=db)
        results = await retriever.retrieve("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_embedding_failure_returns_empty(self):
        from backend.rag.retriever import Retriever
        from backend.vectordb.qdrant_adapter import QdrantAdapter
        from backend.rag.embedder import Embedder

        mock_embedder = MagicMock(spec=Embedder)
        mock_embedder.embed_one = AsyncMock(side_effect=RuntimeError("embedding failed"))
        db = QdrantAdapter(in_memory=True)

        retriever = Retriever(embedder=mock_embedder, vector_db=db)
        results = await retriever.retrieve("query")
        assert results == []  # Fail gracefully