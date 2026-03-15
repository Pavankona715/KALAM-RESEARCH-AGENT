"""
LLM Router
==========
Intelligent routing between LLM providers/models.

Strategies:
1. Fallback: Try primary model, fall back to secondary on failure
2. Cost-aware: Route cheap tasks to cheaper models
3. Capability-aware: Route tool-heavy tasks to capable models

This is where you'd add:
- A/B testing different models
- Budget enforcement (stop using GPT-4 when daily budget exceeded)
- Latency-based routing (use faster model during high traffic)
- Provider health tracking (stop routing to a provider that's failing)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from backend.llm.base import (
    LLMError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
)
from backend.llm.litellm_provider import LiteLLMProvider
from backend.observability.logger import get_logger

logger = get_logger(__name__)


class TaskComplexity(str, Enum):
    """Classify task complexity to guide model selection."""
    SIMPLE = "simple"      # Short Q&A, classification → cheap/fast model
    STANDARD = "standard"  # Typical chat, summarization → default model
    COMPLEX = "complex"    # Multi-step reasoning, code → most capable model


@dataclass
class ModelConfig:
    """Configuration for a single model in the routing table."""
    model: str
    max_tokens: int = 4096
    supports_tools: bool = True
    supports_vision: bool = False
    cost_per_1k_input: float = 0.0   # USD
    cost_per_1k_output: float = 0.0


# Model capability/cost reference table
MODEL_CONFIGS: dict[str, ModelConfig] = {
    "gpt-4o": ModelConfig(
        model="gpt-4o",
        max_tokens=16384,
        supports_tools=True,
        supports_vision=True,
        cost_per_1k_input=0.0025,
        cost_per_1k_output=0.01,
    ),
    "gpt-4o-mini": ModelConfig(
        model="gpt-4o-mini",
        max_tokens=16384,
        supports_tools=True,
        supports_vision=True,
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
    ),
    "claude-sonnet-4-5": ModelConfig(
        model="claude-sonnet-4-5",
        max_tokens=8096,
        supports_tools=True,
        supports_vision=True,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
    ),
    "groq/llama3-70b-8192": ModelConfig(
        model="groq/llama3-70b-8192",
        max_tokens=8192,
        supports_tools=True,
        supports_vision=False,
        cost_per_1k_input=0.00059,
        cost_per_1k_output=0.00079,
    ),
}


class LLMRouter:
    """
    Routes LLM requests to appropriate models with fallback support.

    Usage:
        router = LLMRouter(
            primary="gpt-4o",
            fallbacks=["claude-sonnet-4-5", "groq/llama3-70b-8192"],
        )
        response = await router.complete(request)

    The router tries the primary model first. On rate limit or unavailability,
    it automatically tries each fallback in order.
    """

    def __init__(
        self,
        primary: Optional[str] = None,
        fallbacks: Optional[list[str]] = None,
    ):
        from backend.config.settings import get_settings
        settings = get_settings()

        self._primary = primary or settings.default_llm_model
        self._fallbacks = fallbacks or []
        self._provider = LiteLLMProvider()

        logger.info(
            "llm_router_initialized",
            primary=self._primary,
            fallbacks=self._fallbacks,
        )

    async def complete(
        self,
        request: LLMRequest,
        *,
        complexity: TaskComplexity = TaskComplexity.STANDARD,
    ) -> LLMResponse:
        """
        Complete a request with automatic fallback.

        Args:
            request: The LLM request
            complexity: Hint for model selection (future: route simple tasks to cheaper models)
        """
        models_to_try = [self._primary] + self._fallbacks

        last_error: Optional[LLMError] = None

        for i, model in enumerate(models_to_try):
            is_fallback = i > 0

            if is_fallback:
                logger.warning(
                    "llm_fallback",
                    from_model=models_to_try[i - 1],
                    to_model=model,
                    reason=str(last_error),
                )

            try:
                # Override model in request
                request.model = model
                response = await self._provider.complete(request)

                if is_fallback:
                    logger.info(
                        "llm_fallback_succeeded",
                        model=model,
                        original_model=models_to_try[0],
                    )

                return response

            except (LLMRateLimitError, LLMProviderUnavailableError) as e:
                last_error = e
                if i < len(models_to_try) - 1:
                    continue  # Try next fallback
                raise  # All fallbacks exhausted

            except LLMError:
                raise  # Non-retryable errors (auth, context length) — don't fallback

        # Should never reach here, but satisfy type checker
        raise last_error or LLMError("All models failed")

    def select_model_for_task(
        self,
        complexity: TaskComplexity,
        requires_tools: bool = False,
        requires_vision: bool = False,
    ) -> str:
        """
        Select the best model for a given task.
        Future: implement cost-aware routing here.
        Currently returns the primary model.
        """
        return self._primary

    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate the USD cost of an LLM call."""
        config = MODEL_CONFIGS.get(model)
        if not config:
            return 0.0

        input_cost = (prompt_tokens / 1000) * config.cost_per_1k_input
        output_cost = (completion_tokens / 1000) * config.cost_per_1k_output
        return round(input_cost + output_cost, 6)


# ─── Dependency injection factory ─────────────────────────────────────────────

_router_instance: Optional[LLMRouter] = None


def get_llm_router() -> LLMRouter:
    """
    Get or create the global LLM router.
    Used as a FastAPI dependency and by agents/tools.

    Usage in route:
        @router.post("/chat")
        async def chat(llm: LLMRouter = Depends(get_llm_router)):
            ...
    """
    global _router_instance
    if _router_instance is None:
        from backend.config.settings import get_settings
        settings = get_settings()
        _router_instance = LLMRouter(
            primary=settings.default_llm_model,
            # Add configured fallbacks — you'd read these from settings
        )
    return _router_instance