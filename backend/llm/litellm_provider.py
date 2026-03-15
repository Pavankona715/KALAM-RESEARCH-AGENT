"""
LiteLLM Provider
================
Single adapter that handles OpenAI, Claude, Gemini, Groq and 100+ other providers
through LiteLLM's unified interface.

Why LiteLLM over direct SDK calls?
- One retry/error handling implementation for all providers
- Automatic cost tracking
- Consistent token counting regardless of provider
- Streaming works identically across providers
- Built-in LangSmith integration

Retry strategy (tenacity):
- Rate limits (429): exponential backoff, up to 3 retries
- Server errors (5xx): exponential backoff, up to 3 retries
- Auth errors (401): no retry (fail immediately — won't self-heal)
- Context length (400): no retry (need to truncate input first)
"""

from __future__ import annotations

import time
from typing import Any, AsyncGenerator, Optional

import litellm
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.config.settings import get_settings
from backend.llm.base import (
    LLMAuthError,
    LLMContextLengthError,
    LLMError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    StreamChunk,
    ToolCall,
    TokenUsage,
)
from backend.observability.logger import get_logger
from backend.observability.metrics import LLMCallMetrics, metrics_collector

logger = get_logger(__name__)


class LiteLLMProvider:
    """
    Unified LLM provider using LiteLLM.

    Usage:
        provider = LiteLLMProvider()
        response = await provider.complete(LLMRequest(
            messages=[Message(role="user", content="Hello")],
            model="gpt-4o",
        ))
        print(response.content)

    Streaming:
        async for chunk in provider.stream(request):
            print(chunk.content, end="", flush=True)
    """

    def __init__(
        self,
        default_model: Optional[str] = None,
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
    ):
        settings = get_settings()

        self._default_model = default_model or settings.default_llm_model
        self._default_temperature = (
            default_temperature
            if default_temperature is not None
            else settings.default_llm_temperature
        )
        self._default_max_tokens = default_max_tokens or settings.default_llm_max_tokens

        # Configure LiteLLM
        self._configure_litellm(settings)

        logger.info(
            "llm_provider_initialized",
            default_model=self._default_model,
            providers=settings.configured_llm_providers,
        )

    def _configure_litellm(self, settings) -> None:
        """Set up LiteLLM with API keys and global configuration."""
        # API keys — LiteLLM reads these from environment automatically,
        # but we set them explicitly to be safe
        if settings.openai_api_key:
            litellm.openai_key = settings.openai_api_key
        if settings.anthropic_api_key:
            litellm.anthropic_key = settings.anthropic_api_key

        # Global settings
        litellm.drop_params = True       # Ignore unsupported params per provider
        litellm.set_verbose = False      # Don't pollute logs with LiteLLM internals

        # Enable LangSmith tracing if configured
        if settings.langchain_tracing_v2 and settings.langchain_api_key:
            litellm.success_callback = ["langsmith"]

    def _build_params(self, request: LLMRequest) -> dict[str, Any]:
        """Convert LLMRequest into LiteLLM API call parameters."""
        params: dict[str, Any] = {
            "model": request.model or self._default_model,
            "messages": [m.to_dict() for m in request.messages],
            "temperature": (
                request.temperature
                if request.temperature is not None
                else self._default_temperature
            ),
            "max_tokens": request.max_tokens or self._default_max_tokens,
        }

        # Tool calling
        if request.tools:
            params["tools"] = [t.to_openai_format() for t in request.tools]
            params["tool_choice"] = request.tool_choice

        # Structured output (JSON mode)
        if request.response_format:
            params["response_format"] = request.response_format

        # LangSmith metadata
        if request.metadata:
            params["metadata"] = request.metadata

        return params

    @staticmethod
    def _parse_tool_calls(response) -> list[ToolCall]:
        """Extract tool calls from the LiteLLM response object."""
        tool_calls = []
        message = response.choices[0].message

        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return tool_calls

        for tc in message.tool_calls:
            try:
                import json
                arguments = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                arguments = {}

            tool_calls.append(ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=arguments,
            ))

        return tool_calls

    @staticmethod
    def _extract_usage(response) -> TokenUsage:
        """Safely extract token counts from response."""
        try:
            usage = response.usage
            return TokenUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
            )
        except AttributeError:
            return TokenUsage()

    @staticmethod
    def _translate_error(e: Exception, model: str) -> LLMError:
        """Map LiteLLM/provider exceptions to our domain exceptions."""
        error_str = str(e).lower()

        if "rate limit" in error_str or "429" in error_str:
            return LLMRateLimitError(str(e), model=model)
        elif "401" in error_str or "authentication" in error_str or "api key" in error_str:
            return LLMAuthError(str(e), model=model)
        elif "context" in error_str or "token" in error_str and "limit" in error_str:
            return LLMContextLengthError(str(e), model=model)
        elif "503" in error_str or "unavailable" in error_str or "overloaded" in error_str:
            return LLMProviderUnavailableError(str(e), model=model)
        else:
            return LLMError(str(e), model=model)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Make a single LLM API call. Retries on rate limits and server errors.
        Raises LLMError subclasses on non-retryable failures.
        """
        params = self._build_params(request)
        model = params["model"]
        start = time.perf_counter()

        try:
            response = await self._complete_with_retry(params)
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            translated = self._translate_error(e, model)

            metrics_collector.record_llm_call(LLMCallMetrics(
                model=model,
                latency_ms=latency_ms,
                success=False,
                error=str(translated),
            ))

            logger.error(
                "llm_call_failed",
                model=model,
                error=str(e),
                latency_ms=round(latency_ms, 2),
            )
            raise translated from e

        latency_ms = (time.perf_counter() - start) * 1000
        usage = self._extract_usage(response)
        tool_calls = self._parse_tool_calls(response)

        content = ""
        if response.choices[0].message.content:
            content = response.choices[0].message.content

        finish_reason = response.choices[0].finish_reason or "stop"

        # Record metrics
        metrics_collector.record_llm_call(LLMCallMetrics(
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            latency_ms=latency_ms,
            success=True,
        ))

        logger.debug(
            "llm_call_complete",
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            latency_ms=round(latency_ms, 2),
            finish_reason=finish_reason,
            has_tool_calls=len(tool_calls) > 0,
        )

        return LLMResponse(
            content=content,
            usage=usage,
            model=model,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
        )

    @retry(
        retry=retry_if_exception_type((
            litellm.RateLimitError,
            litellm.ServiceUnavailableError,
            litellm.Timeout,
        )),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _complete_with_retry(self, params: dict) -> Any:
        """Inner call with retry logic. Decorated separately for clean tenacity config."""
        return await litellm.acompletion(**params)

    async def stream(
        self, request: LLMRequest
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream the response token by token.

        Usage:
            full_text = ""
            async for chunk in provider.stream(request):
                full_text += chunk.content
                if chunk.finish_reason:
                    print(f"Done. Tokens: {chunk.usage.total_tokens}")
        """
        params = self._build_params(request)
        params["stream"] = True

        try:
            response = await litellm.acompletion(**params)

            async for chunk in response:
                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

                content = ""
                if hasattr(delta, "content") and delta.content:
                    content = delta.content

                usage = None
                if finish_reason and hasattr(chunk, "usage") and chunk.usage:
                    usage = self._extract_usage(chunk)

                yield StreamChunk(
                    content=content,
                    finish_reason=finish_reason,
                    usage=usage,
                )

        except Exception as e:
            raise self._translate_error(e, params["model"]) from e

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.
        Uses the embedding model configured in settings.
        """
        settings = get_settings()
        start = time.perf_counter()

        try:
            response = await litellm.aembedding(
                model=settings.embedding_model,
                input=texts,
            )
            latency_ms = (time.perf_counter() - start) * 1000

            logger.debug(
                "embedding_complete",
                model=settings.embedding_model,
                text_count=len(texts),
                latency_ms=round(latency_ms, 2),
            )

            return [item["embedding"] for item in response.data]

        except Exception as e:
            raise self._translate_error(e, settings.embedding_model) from e

    def get_model_name(self) -> str:
        return self._default_model