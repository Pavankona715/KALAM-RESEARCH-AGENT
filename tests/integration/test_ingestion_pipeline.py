"""
Document Ingestion Pipeline Tests
==================================
Tests for all parsers and the full ingestion pipeline.

- Parser tests: use real files (created in tmp_path)
- Pipeline tests: mock RAG pipeline and DB
- No network calls, no real LLM/embedding API calls
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# ─── TXT Parser Tests ─────────────────────────────────────────────────────────

class TestTxtParser:
    @pytest.mark.asyncio
    async def test_reads_utf8_file(self, tmp_path):
        from backend.ingestion.parsers.txt_parser import parse_txt
        f = tmp_path / "test.txt"
        f.write_text("Hello world, this is UTF-8 text.", encoding="utf-8")
        result = await parse_txt(f)
        assert "Hello world" in result

    @pytest.mark.asyncio
    async def test_reads_markdown_file(self, tmp_path):
        from backend.ingestion.parsers.txt_parser import parse_txt
        f = tmp_path / "test.md"
        f.write_text("# Heading\n\nParagraph text here.", encoding="utf-8")
        result = await parse_txt(f)
        assert "Heading" in result
        assert "Paragraph text" in result

    @pytest.mark.asyncio
    async def test_handles_latin1_encoding(self, tmp_path):
        from backend.ingestion.parsers.txt_parser import parse_txt
        f = tmp_path / "latin.txt"
        f.write_bytes("caf\xe9 and r\xe9sum\xe9".encode("latin-1"))
        result = await parse_txt(f)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_empty_file_returns_empty_string(self, tmp_path):
        from backend.ingestion.parsers.txt_parser import parse_txt
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        result = await parse_txt(f)
        assert result == ""


# ─── HTML Parser Tests ────────────────────────────────────────────────────────

class TestHtmlParser:
    @pytest.mark.asyncio
    async def test_extracts_paragraph_text(self, tmp_path):
        from backend.ingestion.parsers.url_parser import parse_html_file
        f = tmp_path / "test.html"
        f.write_text(
            "<html><body>"
            "<p>This is the main content paragraph with enough text to pass the filter.</p>"
            "<p>Another paragraph with sufficient content here.</p>"
            "</body></html>",
            encoding="utf-8"
        )
        result = await parse_html_file(f)
        assert "main content paragraph" in result

    @pytest.mark.asyncio
    async def test_strips_script_tags(self, tmp_path):
        from backend.ingestion.parsers.url_parser import parse_html_file
        f = tmp_path / "test.html"
        f.write_text(
            "<html><body>"
            "<script>var x = 'should not appear';</script>"
            "<p>This is the real content that should appear in output text.</p>"
            "</body></html>",
            encoding="utf-8"
        )
        result = await parse_html_file(f)
        assert "should not appear" not in result
        assert "real content" in result

    @pytest.mark.asyncio
    async def test_strips_style_tags(self, tmp_path):
        from backend.ingestion.parsers.url_parser import parse_html_file
        f = tmp_path / "test.html"
        f.write_text(
            "<html><head><style>.cls { color: red; }</style></head>"
            "<body><p>Visible content that should appear in the extracted text output.</p></body></html>",
            encoding="utf-8"
        )
        result = await parse_html_file(f)
        assert "color: red" not in result
        assert "Visible content" in result


# ─── URL Parser Tests ─────────────────────────────────────────────────────────

class TestUrlParser:
    @pytest.mark.asyncio
    async def test_parse_url_success(self):
        from backend.ingestion.parsers.url_parser import parse_url

        html_content = (
            "<html><body>"
            "<p>This is article content with sufficient length to pass filters.</p>"
            "<p>Second paragraph with more meaningful content here.</p>"
            "</body></html>"
        )

        with patch("httpx.AsyncClient") as MockClient:
            mock_response = MagicMock()
            mock_response.text = html_content
            mock_response.headers = {"content-type": "text/html"}
            mock_response.raise_for_status = MagicMock()

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client

            result = await parse_url("https://example.com/article")

        assert "article content" in result

    @pytest.mark.asyncio
    async def test_parse_url_rejects_non_html(self):
        from backend.ingestion.parsers.url_parser import parse_url

        with patch("httpx.AsyncClient") as MockClient:
            mock_response = MagicMock()
            mock_response.headers = {"content-type": "application/pdf"}
            mock_response.raise_for_status = MagicMock()

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client

            with pytest.raises(ValueError, match="non-text content type"):
                await parse_url("https://example.com/file.pdf")


# ─── parse_document Router Tests ──────────────────────────────────────────────

class TestParseDocument:
    @pytest.mark.asyncio
    async def test_routes_txt_to_txt_parser(self, tmp_path):
        from backend.ingestion.parsers import parse_document
        f = tmp_path / "file.txt"
        f.write_text("Plain text content for routing test.", encoding="utf-8")
        result = await parse_document(f)
        assert "Plain text content" in result

    @pytest.mark.asyncio
    async def test_routes_md_to_txt_parser(self, tmp_path):
        from backend.ingestion.parsers import parse_document
        f = tmp_path / "file.md"
        f.write_text("# Title\n\nMarkdown content here.", encoding="utf-8")
        result = await parse_document(f)
        assert "Markdown content" in result

    @pytest.mark.asyncio
    async def test_raises_for_unsupported_extension(self, tmp_path):
        from backend.ingestion.parsers import parse_document
        f = tmp_path / "file.xyz"
        f.write_text("content")
        with pytest.raises(ValueError, match="Unsupported file type"):
            await parse_document(f)

    @pytest.mark.asyncio
    async def test_raises_for_missing_file(self, tmp_path):
        from backend.ingestion.parsers import parse_document
        with pytest.raises(FileNotFoundError):
            await parse_document(tmp_path / "nonexistent.txt")

    @pytest.mark.asyncio
    async def test_routes_url_string(self):
        from backend.ingestion.parsers import parse_document
        from backend.ingestion.parsers.url_parser import parse_url

        with patch("backend.ingestion.parsers.url_parser.parse_url",
                   new_callable=lambda: lambda *a, **kw: AsyncMock(return_value="URL content")()
                   ) as mock_parse:
            with patch("backend.ingestion.parsers.parse_url",
                       new_callable=AsyncMock, return_value="URL content") as mock_url:
                result = await parse_document("https://example.com", is_url=True)
            assert result == "URL content"


# ─── DocumentIngestionPipeline Tests ─────────────────────────────────────────

class TestDocumentIngestionPipeline:
    def _make_pipeline(self):
        from backend.ingestion.pipeline import DocumentIngestionPipeline
        from backend.rag.pipeline import RAGPipeline, IngestionResult

        mock_rag = MagicMock(spec=RAGPipeline)
        mock_rag.ingest_text = AsyncMock(return_value=IngestionResult(
            doc_id="doc-1",
            filename="test.txt",
            chunks_created=3,
            chunks_embedded=3,
            chunks_stored=3,
            success=True,
        ))

        pipeline = DocumentIngestionPipeline(rag_pipeline=mock_rag)
        # Patch DB update to avoid needing a real DB
        pipeline._update_db_status = AsyncMock()
        return pipeline, mock_rag

    @pytest.mark.asyncio
    async def test_ingest_txt_file_success(self, tmp_path):
        pipeline, mock_rag = self._make_pipeline()

        f = tmp_path / "test.txt"
        f.write_text("This is test content. " * 20, encoding="utf-8")

        result = await pipeline.ingest_file(
            file_path=f,
            doc_id="doc-1",
            filename="test.txt",
            user_id="user-1",
        )

        assert result.success is True
        assert result.chunks_stored == 3
        assert result.text_length > 0
        mock_rag.ingest_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_empty_file_fails_gracefully(self, tmp_path):
        pipeline, mock_rag = self._make_pipeline()

        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")

        result = await pipeline.ingest_file(
            file_path=f,
            doc_id="doc-empty",
            filename="empty.txt",
            user_id="user-1",
        )

        assert result.success is False
        assert result.error is not None
        mock_rag.ingest_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_missing_file_fails_gracefully(self, tmp_path):
        pipeline, mock_rag = self._make_pipeline()

        result = await pipeline.ingest_file(
            file_path=tmp_path / "nonexistent.pdf",
            doc_id="doc-missing",
            filename="nonexistent.pdf",
            user_id="user-1",
        )

        assert result.success is False
        assert "Parse failed" in result.error

    @pytest.mark.asyncio
    async def test_ingest_url_success(self):
        pipeline, mock_rag = self._make_pipeline()

        with patch("backend.ingestion.parsers.url_parser.parse_url",
                   new_callable=AsyncMock,
                   return_value="Web page content. " * 20) as mock_parse:

            result = await pipeline.ingest_url(
                url="https://example.com/article",
                doc_id="doc-url",
                user_id="user-1",
            )

        assert result.success is True
        assert result.chunks_stored == 3

    @pytest.mark.asyncio
    async def test_ingest_url_fetch_failure(self):
        pipeline, mock_rag = self._make_pipeline()

        with patch("backend.ingestion.parsers.url_parser.parse_url",
                   new_callable=AsyncMock,
                   side_effect=ValueError("HTTP 404")):

            result = await pipeline.ingest_url(
                url="https://example.com/notfound",
                doc_id="doc-404",
                user_id="user-1",
            )

        assert result.success is False
        assert "URL fetch failed" in result.error

    @pytest.mark.asyncio
    async def test_db_status_updated_on_success(self, tmp_path):
        pipeline, _ = self._make_pipeline()
        f = tmp_path / "doc.txt"
        f.write_text("Content. " * 30)

        await pipeline.ingest_file(
            file_path=f, doc_id="doc-1",
            filename="doc.txt", user_id="user-1"
        )

        # DB update should have been called with 'ready'
        pipeline._update_db_status.assert_called_with(
            "doc-1", "ready", chunk_count=3, error=None
        )

    @pytest.mark.asyncio
    async def test_db_update_failure_does_not_abort_result(self, tmp_path):
        """If DB update fails, the ingestion result is still returned."""
        pipeline, _ = self._make_pipeline()
        pipeline._update_db_status = AsyncMock(side_effect=RuntimeError("DB down"))

        f = tmp_path / "doc.txt"
        f.write_text("Content. " * 30)

        result = await pipeline.ingest_file(
            file_path=f, doc_id="doc-1",
            filename="doc.txt", user_id="user-1"
        )
        # Should still succeed even though DB update failed
        assert result.success is True