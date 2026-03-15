"""
LLM Layer Unit Tests
====================
Tests for LLMRequest/Response data types, LiteLLMProvider behavior,
router fallback logic, and context builder assembly.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.llm.base import (
    LLMRequest,
    LLMResponse,
    Message,
    ToolCall,
    ToolDefinition,
    TokenUsage,
    LLMRateLimitError,
    LLMAuthError,
)
from backend.context.builder import (
    ContextBuilder,
    ContextConfig,
    ContextInput,
    RetrievedDocument,
    estimate_tokens,
)


# ─── Data Type Tests ──────────────────────────────────────────────────────────

class TestTokenUsage:
    def test_total_tokens(self):
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        assert usage.total_tokens == 150

    def test_addition(self):
        a = TokenUsage(prompt_tokens=100, completion_tokens=50)
        b = TokenUsage(prompt_tokens=200, completion_tokens=100)
        total = a + b
        assert total.prompt_tokens == 300
        assert total.completion_tokens == 150
        assert total.total_tokens == 450


class TestMessage:
    def test_simple_message_to_dict(self):
        msg = Message(role="user", content="Hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Hello"
        assert "tool_call_id" not in d

    def test_tool_result_message(self):
        msg = Message(role="tool", content="result", tool_call_id="call_123")
        d = msg.to_dict()
        assert d["tool_call_id"] == "call_123"


class TestToolDefinition:
    def test_to_openai_format(self):
        tool = ToolDefinition(
            name="web_search",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        fmt = tool.to_openai_format()
        assert fmt["type"] == "function"
        assert fmt["function"]["name"] == "web_search"
        assert "parameters" in fmt["function"]


class TestLLMResponse:
    def test_has_tool_calls_false(self):
        resp = LLMResponse(
            content="Hello",
            usage=TokenUsage(),
            model="gpt-4o",
        )
        assert resp.has_tool_calls is False

    def test_has_tool_calls_true(self):
        resp = LLMResponse(
            content="",
            usage=TokenUsage(),
            model="gpt-4o",
            tool_calls=[ToolCall(id="1", name="search", arguments={})],
        )
        assert resp.has_tool_calls is True

    def test_is_complete_stop(self):
        resp = LLMResponse(
            content="Done",
            usage=TokenUsage(),
            model="gpt-4o",
            finish_reason="stop",
        )
        assert resp.is_complete is True

    def test_is_complete_length_false(self):
        resp = LLMResponse(
            content="Truncat...",
            usage=TokenUsage(),
            model="gpt-4o",
            finish_reason="length",
        )
        assert resp.is_complete is False


# ─── Context Builder Tests ────────────────────────────────────────────────────

class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 1  # minimum 1

    def test_rough_estimate(self):
        text = "a" * 400   # 400 chars ≈ 100 tokens
        assert estimate_tokens(text) == 100


class TestContextBuilder:
    def setup_method(self):
        self.builder = ContextBuilder(config=ContextConfig(
            max_tokens=4000,
            history_max_messages=10,
        ))

    def test_builds_basic_context(self):
        ctx = self.builder.build(ContextInput(
            user_message="What is AI?",
        ))
        assert len(ctx.messages) >= 2  # system + user
        assert ctx.messages[-1].role == "user"
        assert ctx.messages[-1].content == "What is AI?"
        assert ctx.messages[0].role == "system"

    def test_includes_conversation_history(self):
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        ctx = self.builder.build(ContextInput(
            user_message="Follow up question",
            conversation_history=history,
        ))
        # Should have system + 2 history + current user = 4 messages
        assert len(ctx.messages) == 4

    def test_includes_retrieved_docs(self):
        docs = [
            RetrievedDocument(
                content="RAG stands for Retrieval Augmented Generation",
                source="paper.pdf",
                score=0.92,
                doc_id="doc_1",
            )
        ]
        ctx = self.builder.build(ContextInput(
            user_message="What is RAG?",
            retrieved_docs=docs,
        ))
        assert "doc_1" in ctx.retrieved_doc_ids
        # RAG section should be in a system message
        rag_content = " ".join(m.content for m in ctx.messages if m.role == "system")
        assert "Retrieved Knowledge" in rag_content

    def test_filters_low_score_docs(self):
        docs = [
            RetrievedDocument(
                content="High relevance content",
                source="good.pdf",
                score=0.9,
                doc_id="good_doc",
            ),
            RetrievedDocument(
                content="Low relevance content",
                source="bad.pdf",
                score=0.2,   # Below default threshold of 0.5
                doc_id="bad_doc",
            ),
        ]
        ctx = self.builder.build(ContextInput(
            user_message="Test",
            retrieved_docs=docs,
        ))
        assert "good_doc" in ctx.retrieved_doc_ids
        assert "bad_doc" not in ctx.retrieved_doc_ids

    def test_history_trimmed_when_too_long(self):
        # Create 30 messages, but builder is configured for max 10
        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
            for i in range(30)
        ]
        ctx = self.builder.build(ContextInput(
            user_message="Current",
            conversation_history=history,
        ))
        assert ctx.truncated is True
        # Should have kept most recent 10 (+ system + current user)
        history_messages = [m for m in ctx.messages if m.role in ("user", "assistant")]
        assert len(history_messages) <= 11  # 10 history + 1 current


# ─── LiteLLM Provider Tests ──────────────────────────────────────────────────

class TestLiteLLMProvider:
    @pytest.mark.asyncio
    async def test_complete_success(self):
        """Test successful LLM call with mocked LiteLLM."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.model = "gpt-4o"

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = mock_response

            from backend.llm.litellm_provider import LiteLLMProvider
            provider = LiteLLMProvider(default_model="gpt-4o")

            request = LLMRequest(
                messages=[Message(role="user", content="Hello")],
            )
            response = await provider.complete(request)

        assert response.content == "Test response"
        assert response.usage.total_tokens == 15
        assert response.finish_reason == "stop"
        assert not response.has_tool_calls

    @pytest.mark.asyncio
    async def test_rate_limit_raises_correct_exception(self):
        """Rate limit errors should be translated to LLMRateLimitError."""
        import litellm

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = litellm.RateLimitError(
                message="Rate limit exceeded",
                llm_provider="openai",
                model="gpt-4o",
            )

            from backend.llm.litellm_provider import LiteLLMProvider
            provider = LiteLLMProvider(default_model="gpt-4o")

            with pytest.raises(LLMRateLimitError):
                await provider.complete(LLMRequest(
                    messages=[Message(role="user", content="Hello")]
                ))


# ─── Router Fallback Tests ────────────────────────────────────────────────────

class TestLLMRouter:
    @pytest.mark.asyncio
    async def test_uses_primary_model(self, mock_llm_response):
        """Primary model is used on the first call when no errors occur."""
        from backend.llm.router import LLMRouter
        from backend.llm.litellm_provider import LiteLLMProvider

        # Instantiate router, then replace the instance-level _provider
        router = LLMRouter(primary="gpt-4o", fallbacks=["gpt-4o-mini"])

        mock_provider = MagicMock(spec=LiteLLMProvider)
        mock_provider.complete = AsyncMock(return_value=mock_llm_response)
        router._provider = mock_provider   # Direct instance assignment — no patch.object needed

        request = LLMRequest(messages=[Message(role="user", content="Hello")])
        response = await router.complete(request)

        assert response.content == mock_llm_response.content
        # Router should have set the model on the request before calling provider
        assert request.model == "gpt-4o"
        mock_provider.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_falls_back_on_rate_limit(self, mock_llm_response):
        """When primary hits rate limit, should automatically try fallback model."""
        from backend.llm.router import LLMRouter
        from backend.llm.litellm_provider import LiteLLMProvider

        call_count = 0

        async def selective_complete(req):
            nonlocal call_count
            call_count += 1
            if req.model == "gpt-4o":
                raise LLMRateLimitError("Rate limited", model="gpt-4o")
            return mock_llm_response

        router = LLMRouter(primary="gpt-4o", fallbacks=["gpt-4o-mini"])
        mock_provider = MagicMock(spec=LiteLLMProvider)
        mock_provider.complete = selective_complete
        router._provider = mock_provider

        request = LLMRequest(messages=[Message(role="user", content="Hello")])
        response = await router.complete(request)

        assert response.content == mock_llm_response.content
        assert call_count == 2  # tried gpt-4o (failed) + gpt-4o-mini (succeeded)

    @pytest.mark.asyncio
    async def test_raises_when_all_fallbacks_exhausted(self):
        """Should re-raise the last error when all models are rate-limited."""
        from backend.llm.router import LLMRouter
        from backend.llm.litellm_provider import LiteLLMProvider

        async def always_rate_limit(req):
            raise LLMRateLimitError("Rate limited", model=req.model)

        router = LLMRouter(primary="gpt-4o", fallbacks=["gpt-4o-mini"])
        mock_provider = MagicMock(spec=LiteLLMProvider)
        mock_provider.complete = always_rate_limit
        router._provider = mock_provider

        with pytest.raises(LLMRateLimitError):
            await router.complete(
                LLMRequest(messages=[Message(role="user", content="Hello")])
            )