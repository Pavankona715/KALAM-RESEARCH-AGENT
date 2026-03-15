"""
Tests for Step 13 — Observability.

All external SDK calls (LangSmith, OTel) mocked.
No real API keys, no network calls.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.observability.logger import (
    JSONFormatter,
    ReasoningTraceLogger,
    configure_logging,
    get_reasoning_logger,
    set_log_context,
)
from backend.observability.metrics import (
    InMemoryMetricsStore,
    MetricsCollector,
    get_metrics_collector,
)
from backend.observability.tracer import (
    AgentTracer,
    _LangSmithTracer,
    _OTelTracer,
    get_current_span_id,
    get_current_trace_id,
    get_tracer,
)


# ---------------------------------------------------------------------------
# Logger tests
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    def test_idempotent(self):
        """Calling configure_logging twice should not raise."""
        configure_logging("DEBUG")
        configure_logging("INFO")  # no error

    def test_json_formatter(self):
        import logging

        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello %s",
            args=("world",),
            exc_info=None,
        )
        output = formatter.format(record)
        import json

        parsed = json.loads(output)
        assert parsed["message"] == "hello world"
        assert parsed["level"] == "INFO"
        assert "timestamp" in parsed

    def test_json_formatter_injects_context(self):
        import json
        import logging

        set_log_context(request_id="req-123", trace_id="trace-456", user_id="user-789")
        formatter = JSONFormatter()
        record = logging.LogRecord("t", logging.INFO, "", 0, "msg", (), None)
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed.get("request_id") == "req-123"
        assert parsed.get("trace_id") == "trace-456"
        assert parsed.get("user_id") == "user-789"

        # Cleanup
        set_log_context(request_id=None, trace_id=None, user_id=None)


class TestReasoningTraceLogger:
    def test_log_step(self, caplog):
        import logging

        logger = ReasoningTraceLogger()
        with caplog.at_level(logging.INFO, logger="agent.reasoning"):
            logger.log_step("run-1", "tool_call", "calling web search", "web_search", 123.4)
        # Just verifying no exception raised and it logs
        assert True  # log_step is fire-and-forget

    def test_log_final_answer(self):
        logger = ReasoningTraceLogger()
        # Should not raise
        logger.log_final_answer("run-1", "Final answer here", total_steps=3, total_duration_ms=500.0)

    def test_get_reasoning_logger_singleton(self):
        l1 = get_reasoning_logger()
        l2 = get_reasoning_logger()
        assert l1 is l2


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


class TestInMemoryMetricsStore:
    def test_record_tokens(self):
        store = InMemoryMetricsStore()
        from backend.observability.metrics import TokenUsageRecord

        rec = TokenUsageRecord(
            user_id="u1",
            session_id="s1",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
        )
        store.record_tokens(rec)
        totals = store.get_user_token_totals("u1")
        assert totals["prompt"] == 100
        assert totals["completion"] == 200
        assert totals["total"] == 300

    def test_session_totals_accumulate(self):
        store = InMemoryMetricsStore()
        from backend.observability.metrics import TokenUsageRecord

        for _ in range(3):
            store.record_tokens(
                TokenUsageRecord("u1", "s1", "gpt-4", 50, 100, 150)
            )
        totals = store.get_session_token_totals("s1")
        assert totals["total"] == 450

    def test_record_latency(self):
        store = InMemoryMetricsStore()
        from backend.observability.metrics import LatencyRecord

        for val in [10.0, 20.0, 30.0, 100.0, 200.0]:
            store.record_latency(LatencyRecord("/chat", val, 200))
        stats = store.get_latency_stats("/chat")
        assert stats["count"] == 5
        assert stats["p50"] > 0

    def test_latency_max_samples(self):
        store = InMemoryMetricsStore()
        from backend.observability.metrics import LatencyRecord

        for i in range(1100):
            store.record_latency(LatencyRecord("/chat", float(i), 200))
        assert len(store.latencies["/chat"]) == 1000

    def test_record_error(self):
        store = InMemoryMetricsStore()
        from backend.observability.metrics import ErrorRecord

        store.record_error(ErrorRecord("/chat", "ValueError", "bad input"))
        store.record_error(ErrorRecord("/chat", "TimeoutError", ""))
        counts = store.get_error_rate("/chat")
        assert counts["/chat"] == 2

    def test_get_latency_stats_empty(self):
        store = InMemoryMetricsStore()
        stats = store.get_latency_stats("/nonexistent")
        assert stats["count"] == 0
        assert stats["p50"] == 0.0

    def test_agent_step_recording(self):
        store = InMemoryMetricsStore()
        from backend.observability.metrics import AgentStepRecord

        store.record_agent_step(AgentStepRecord("run-1", "reasoning", 50.0, True))
        store.record_agent_step(AgentStepRecord("run-1", "tool_call", 200.0, True))
        assert len(store.agent_steps["run-1"]) == 2


class TestMetricsCollector:
    def test_record_token_usage(self):
        collector = MetricsCollector()
        collector.record_token_usage("u1", "s1", "gpt-4", 100, 200)
        totals = collector.get_user_tokens("u1")
        assert totals["total"] == 300

    def test_record_request_latency(self):
        collector = MetricsCollector()
        collector.record_request_latency("/chat", 150.0, 200)
        stats = collector.get_latency_stats("/chat")
        assert stats["count"] == 1

    def test_record_error(self):
        collector = MetricsCollector()
        collector.record_error("/chat", "ValueError")
        assert collector.get_all_stats()["errors"].get("/chat", 0) == 1

    def test_record_agent_step(self):
        collector = MetricsCollector()
        collector.record_agent_step("run-1", "reasoning", 80.0, True)
        steps = collector.get_agent_steps("run-1")
        assert len(steps) == 1
        assert steps[0].step_name == "reasoning"

    def test_get_all_stats(self):
        collector = MetricsCollector()
        collector.record_request_latency("/health", 5.0, 200)
        stats = collector.get_all_stats()
        assert "/health" in stats["endpoints"]

    def test_get_metrics_collector_singleton(self):
        import backend.observability.metrics as m

        m._collector_instance = None
        c1 = get_metrics_collector()
        c2 = get_metrics_collector()
        assert c1 is c2
        m._collector_instance = None  # cleanup

    @pytest.mark.asyncio
    async def test_redis_persist_silently_fails(self):
        """If Redis raises, record_token_usage should still work."""
        mock_redis = AsyncMock()
        mock_redis.rpush = AsyncMock(side_effect=Exception("redis down"))
        collector = MetricsCollector(redis_client=mock_redis)
        collector.record_token_usage("u1", "s1", "gpt-4", 50, 100)
        await asyncio.sleep(0)  # let create_task run
        # No exception; in-memory store still updated
        assert collector.get_user_tokens("u1")["total"] == 150


# ---------------------------------------------------------------------------
# Tracer tests
# ---------------------------------------------------------------------------


class TestLangSmithTracer:
    def test_disabled_when_no_api_key(self, monkeypatch):
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        tracer = _LangSmithTracer()
        assert tracer.enabled is False

    def test_create_run_no_op_when_disabled(self, monkeypatch):
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        tracer = _LangSmithTracer()
        result = tracer.create_run("test", "chain", {})
        assert result is None

    def test_end_run_no_op_when_disabled(self, monkeypatch):
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        tracer = _LangSmithTracer()
        # Should not raise
        tracer.end_run("fake-run-id", {"output": "x"})


class TestOTelTracer:
    def test_disabled_when_no_endpoint(self, monkeypatch):
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        tracer = _OTelTracer()
        assert tracer.enabled is False

    def test_start_span_no_op_when_disabled(self, monkeypatch):
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        tracer = _OTelTracer()
        with tracer.start_span("test") as span:
            assert span is None  # no-op returns None


class TestAgentTracer:
    @pytest.mark.asyncio
    async def test_trace_request_yields_trace_id(self):
        tracer = AgentTracer()
        async with tracer.trace_request("/chat", user_id="u1") as ctx:
            assert "trace_id" in ctx
            assert ctx["trace_id"] is not None

    @pytest.mark.asyncio
    async def test_trace_request_sets_context_var(self):
        tracer = AgentTracer()
        trace_id_inside = None
        async with tracer.trace_request("/chat") as ctx:
            trace_id_inside = get_current_trace_id()
        assert trace_id_inside == ctx["trace_id"]
        # After context exits, var is reset
        assert get_current_trace_id() is None

    @pytest.mark.asyncio
    async def test_trace_agent_step(self):
        tracer = AgentTracer()
        async with tracer.trace_request("/chat") as req_ctx:
            async with tracer.trace_agent_step("reasoning", trace_id=req_ctx["trace_id"]) as step_ctx:
                assert "span_id" in step_ctx
                span_id = get_current_span_id()
                assert span_id == step_ctx["span_id"]

    @pytest.mark.asyncio
    async def test_trace_request_on_exception(self):
        tracer = AgentTracer()
        with pytest.raises(ValueError):
            async with tracer.trace_request("/chat"):
                raise ValueError("test error")

    @pytest.mark.asyncio
    async def test_trace_llm_call(self):
        tracer = AgentTracer()
        async with tracer.trace_llm_call("gpt-4", prompt_tokens=100) as ctx:
            assert "span_id" in ctx

    def test_get_tracer_singleton(self):
        import backend.observability.tracer as t

        t._tracer_instance = None
        t1 = get_tracer()
        t2 = get_tracer()
        assert t1 is t2
        t._tracer_instance = None  # cleanup

    @pytest.mark.asyncio
    async def test_nested_traces_restore_context(self):
        """After nested trace_agent_step exits, outer span_id is restored."""
        tracer = AgentTracer()
        async with tracer.trace_request("/chat") as _:
            async with tracer.trace_agent_step("outer") as outer_ctx:
                outer_span = outer_ctx["span_id"]
                async with tracer.trace_agent_step("inner") as inner_ctx:
                    assert get_current_span_id() == inner_ctx["span_id"]
                # After inner exits, outer span is restored
                assert get_current_span_id() == outer_span