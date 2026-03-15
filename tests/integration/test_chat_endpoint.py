"""
Chat & Health Endpoint Integration Tests
=========================================
Tests HTTP layer behavior: routing, validation, auth, response shapes.
Uses mock LLM and in-memory SQLite — no real external services needed.
"""

import pytest
from httpx import AsyncClient


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_liveness_returns_200(self, client: AsyncClient):
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "uptime_seconds" in data
        assert "environment" in data

    @pytest.mark.asyncio
    async def test_response_has_request_id_header(self, client: AsyncClient):
        response = await client.get("/health")
        assert "x-request-id" in response.headers

    @pytest.mark.asyncio
    async def test_response_has_timing_header(self, client: AsyncClient):
        response = await client.get("/health")
        assert "x-process-time" in response.headers


class TestChatEndpoint:
    @pytest.mark.asyncio
    async def test_chat_creates_session(self, client: AsyncClient):
        """POST /chat without session_id should create a new session."""
        response = await client.post("/chat", json={
            "message": "What is machine learning?",
        })
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "message" in data
        assert data["message"]["role"] == "assistant"
        assert len(data["message"]["content"]) > 0

    @pytest.mark.asyncio
    async def test_chat_response_has_model_info(self, client: AsyncClient):
        response = await client.post("/chat", json={"message": "Hello"})
        assert response.status_code == 200
        data = response.json()
        assert data["message"]["model"] is not None
        assert data["message"]["tokens_used"] >= 0

    @pytest.mark.asyncio
    async def test_chat_requires_auth(self, app, mock_llm):
        """Requests without API key should return 401."""
        from httpx import ASGITransport, AsyncClient
        from backend.api.dependencies import get_current_user
        from backend.llm.router import get_llm_router

        # Clone app but remove the auth override so auth is enforced
        auth_app = app
        del auth_app.dependency_overrides[get_current_user]
        # Keep the LLM mock so auth check happens before any LLM call
        auth_app.dependency_overrides[get_llm_router] = lambda: mock_llm

        async with AsyncClient(
            transport=ASGITransport(app=auth_app),
            base_url="http://test",
        ) as unauthed_client:
            response = await unauthed_client.post("/chat", json={"message": "Hello"})

        # Restore auth override for other tests
        from unittest.mock import MagicMock
        auth_app.dependency_overrides[get_current_user] = lambda: MagicMock()

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_chat_validates_empty_message(self, client: AsyncClient):
        response = await client.post("/chat", json={"message": ""})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_validates_agent_type(self, client: AsyncClient):
        response = await client.post("/chat", json={
            "message": "Hello",
            "agent_type": "invalid_type",
        })
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self, client: AsyncClient):
        response = await client.get("/chat/sessions")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    @pytest.mark.asyncio
    async def test_session_not_found(self, client: AsyncClient):
        import uuid
        fake_id = uuid.uuid4()
        response = await client.get(f"/chat/sessions/{fake_id}")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_chat_then_list_sessions(self, client: AsyncClient):
        """After creating a chat, it should appear in session list."""
        # Create a chat session
        chat_resp = await client.post("/chat", json={"message": "Hello"})
        assert chat_resp.status_code == 200

        # List sessions
        list_resp = await client.get("/chat/sessions")
        assert list_resp.status_code == 200
        sessions = list_resp.json()
        assert len(sessions) == 1
        assert sessions[0]["id"] == chat_resp.json()["session_id"]

    @pytest.mark.asyncio
    async def test_delete_session(self, client: AsyncClient):
        # Create session
        chat_resp = await client.post("/chat", json={"message": "Hello"})
        session_id = chat_resp.json()["session_id"]

        # Delete it
        del_resp = await client.delete(f"/chat/sessions/{session_id}")
        assert del_resp.status_code == 204

        # Confirm gone
        get_resp = await client.get(f"/chat/sessions/{session_id}")
        assert get_resp.status_code == 404


class TestUploadEndpoint:
    @pytest.mark.asyncio
    async def test_upload_txt_file(self, client: AsyncClient, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test document about machine learning.")

        with open(test_file, "rb") as f:
            response = await client.post(
                "/upload",
                files={"file": ("test.txt", f, "text/plain")},
            )

        assert response.status_code == 202
        data = response.json()
        assert data["filename"] == "test.txt"
        assert data["doc_type"] == "txt"
        assert data["status"] == "pending"

    @pytest.mark.asyncio
    async def test_upload_rejects_invalid_extension(self, client: AsyncClient, tmp_path):
        test_file = tmp_path / "test.exe"
        test_file.write_bytes(b"MZ malicious content")

        with open(test_file, "rb") as f:
            response = await client.post(
                "/upload",
                files={"file": ("test.exe", f, "application/octet-stream")},
            )

        assert response.status_code == 422


class TestSearchEndpoint:
    @pytest.mark.asyncio
    async def test_search_returns_results_structure(self, client: AsyncClient):
        response = await client.post("/search", json={
            "query": "machine learning algorithms",
            "top_k": 3,
        })
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert "total_found" in data
        assert "latency_ms" in data

    @pytest.mark.asyncio
    async def test_search_validates_empty_query(self, client: AsyncClient):
        response = await client.post("/search", json={"query": ""})
        assert response.status_code == 422