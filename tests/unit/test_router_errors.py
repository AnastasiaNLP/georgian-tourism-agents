"""Tests for error handling in api/router.py — exception leak (AF) and eval_node LLM config (AG)."""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException


# ---------------------------------------------------------------------------
# AF — Graph exceptions must not leak tracebacks to the caller
# ---------------------------------------------------------------------------

_FIXED_THREAD_ID = "test-thread-fixed"


@pytest.mark.asyncio
async def test_graph_error_returns_request_id_not_traceback():
    """500 response must contain request_id, no tracebacks, and lock must be released."""
    from api.router import plan_trip, PlanRequest

    req = PlanRequest(query="trip to mountains", language="en")

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(
        side_effect=RuntimeError("internal/module/path.py line 42: secret_var='abc'")
    )

    http_request = MagicMock()
    http_request.app.state.graph = mock_graph

    guard_result = MagicMock()
    guard_result.passed = True

    http_request.app.state.guardrails = MagicMock()
    http_request.app.state.guardrails.check.return_value = guard_result

    mock_release = AsyncMock()
    with patch("api.router._derive_thread_id", return_value=_FIXED_THREAD_ID), \
         patch("api.router._try_acquire_thread", return_value="tok"), \
         patch("api.router._release_thread", mock_release):
        with pytest.raises(HTTPException) as exc_info:
            await plan_trip(req, http_request)

    assert exc_info.value.status_code == 500
    detail = exc_info.value.detail
    assert "request_id" in detail
    assert "internal_error" in str(detail)
    assert "internal/module" not in str(detail)
    assert "secret_var" not in str(detail)
    assert "line 42" not in str(detail)
    mock_release.assert_awaited_once_with(_FIXED_THREAD_ID, "tok")


@pytest.mark.asyncio
async def test_graph_503_when_not_initialized():
    """503 is returned when graph is absent, and lock must still be released."""
    from api.router import plan_trip, PlanRequest

    req = PlanRequest(query="trip to Tbilisi", language="en")

    http_request = MagicMock()
    http_request.app.state.graph = None

    guard_result = MagicMock()
    guard_result.passed = True

    http_request.app.state.guardrails = MagicMock()
    http_request.app.state.guardrails.check.return_value = guard_result

    mock_release = AsyncMock()
    with patch("api.router._derive_thread_id", return_value=_FIXED_THREAD_ID), \
         patch("api.router._try_acquire_thread", return_value="tok"), \
         patch("api.router._release_thread", mock_release):
        with pytest.raises(HTTPException) as exc_info:
            await plan_trip(req, http_request)

    assert exc_info.value.status_code == 503
    mock_release.assert_awaited_once_with(_FIXED_THREAD_ID, "tok")


# ---------------------------------------------------------------------------
# Hard request timeout — a hung graph yields a graceful degraded response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_graph_hard_timeout_returns_degraded():
    """When the graph exceeds the hard ceiling, return degraded (not 500/hang); lock released."""
    from api.router import plan_trip, PlanRequest

    req = PlanRequest(query="trip to mountains", language="en")

    async def _slow_ainvoke(*args, **kwargs):
        await asyncio.sleep(2)
        return {}

    mock_graph = MagicMock()
    mock_graph.ainvoke = _slow_ainvoke

    http_request = MagicMock()
    http_request.app.state.graph = mock_graph
    guard_result = MagicMock()
    guard_result.passed = True
    http_request.app.state.guardrails = MagicMock()
    http_request.app.state.guardrails.check.return_value = guard_result

    small_timeout = MagicMock()
    small_timeout.request_hard_timeout_seconds = 0.05

    mock_release = AsyncMock()
    with patch("api.router._derive_thread_id", return_value=_FIXED_THREAD_ID), \
         patch("api.router._try_acquire_thread", return_value="tok"), \
         patch("api.router._release_thread", mock_release), \
         patch("api.router.get_settings", return_value=small_timeout):
        result = await plan_trip(req, http_request)

    assert result.execution_mode == "degraded"
    assert result.status == "ok"
    assert result.has_itinerary is False
    assert result.response  # non-empty timeout message
    mock_release.assert_awaited_once_with(_FIXED_THREAD_ID, "tok")


def test_thread_lock_ttl_exceeds_hard_timeout():
    """Lock TTL must track the hard request timeout (+margin), not a fixed constant."""
    from api import router

    fake = MagicMock()
    fake.request_hard_timeout_seconds = 600.0
    with patch("api.router.get_settings", return_value=fake):
        ttl = router._thread_lock_ttl()
    assert ttl > 600  # outlives the longest possible request


# ---------------------------------------------------------------------------
# AG — eval_node must pass api_key and timeout to ChatOpenAI
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_eval_node_llm_receives_api_key_and_timeout():
    """eval_node must construct ChatOpenAI with api_key from settings and timeout=60."""
    captured: dict = {}

    class _CaptureLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def ainvoke(self, prompt):
            return MagicMock(
                content='{"realism": 1.0, "distance": 1.0, "feasibility": 1.0, "comment": "ok"}'
            )

    with patch("langchain_openai.ChatOpenAI", _CaptureLLM):
        from graph.main_graph import eval_node
        await eval_node({
            "request_id": "eval-test",
            "feature_flags": {"ENABLE_EVAL": True},
            "raw_itinerary": {
                "days": [{"day": 1, "location": "Tbilisi", "activities": []}],
                "total_days": 1,
                "pace": "moderate",
            },
        })

    assert "api_key" in captured, "eval_node must pass api_key to ChatOpenAI"
    assert captured["api_key"] is not None
    assert captured.get("timeout") == 60, "eval_node must pass timeout=60 to ChatOpenAI"
