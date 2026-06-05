"""
Unit tests for fallback behavior under external service failures.
Covers: safe_node timeout/exception cascade, geo_agent ORS failure,
orchestrator Anthropic failure → _fallback_classification.
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from agents.error_handler import safe_node, AGENT_TIMEOUTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(agent_name: str = "search_agent") -> dict:
    return {
        "request_id": "fallback-test",
        "user_query": "Батуми 2 дня",
        "user_language": "ru",
        "execution_mode": "normal",
        "agent_history": [],
        "errors": [],
        "budget_state": {},
    }


async def _ok_node(state: dict) -> dict:
    return {"agent_history": ["ok_agent"]}


async def _raising_node(state: dict) -> dict:
    raise ConnectionError("Qdrant unreachable")


async def _slow_node(state: dict) -> dict:
    await asyncio.sleep(999)
    return {}


# ---------------------------------------------------------------------------
# safe_node — timeout
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_safe_node_timeout_returns_degraded():
    """Agent that hangs → safe_node returns degraded, no exception raised."""
    wrapped = safe_node(_slow_node, "search_agent")
    # Patch at execution time — wrapper reads AGENT_TIMEOUTS on each call.
    with patch.dict("agents.error_handler.AGENT_TIMEOUTS", {"search_agent": 0.05}):
        result = await wrapped(_base_state())
    assert result["execution_mode"] == "degraded"
    assert any("timeout" in e for e in result["errors"])


@pytest.mark.asyncio
async def test_safe_node_exception_returns_degraded():
    """Agent that raises ConnectionError → safe_node catches, returns degraded."""
    wrapped = safe_node(_raising_node, "search_agent")
    result = await wrapped(_base_state())
    assert result["execution_mode"] == "degraded"
    assert any("Qdrant unreachable" in e for e in result["errors"])


@pytest.mark.asyncio
async def test_safe_node_success_passes_through():
    """Healthy agent → safe_node returns result unchanged."""
    wrapped = safe_node(_ok_node, "search_agent")
    result = await wrapped(_base_state())
    assert result == {"agent_history": ["ok_agent"]}


@pytest.mark.asyncio
async def test_safe_node_transient_error_stays_normal():
    """First transient error in normal mode keeps execution_mode=normal."""
    async def _transient(state: dict) -> dict:
        raise TimeoutError("transient")

    wrapped = safe_node(_transient, "search_agent")
    result = await wrapped(_base_state())
    # TimeoutError is in TRANSIENT_ERRORS — first hit stays normal
    assert result["execution_mode"] == "normal"


@pytest.mark.asyncio
async def test_safe_node_already_degraded_escalates_to_emergency():
    """Exception in degraded mode → escalates to emergency."""
    state = _base_state()
    state["execution_mode"] = "degraded"
    wrapped = safe_node(_raising_node, "search_agent")
    result = await wrapped(state)
    assert result["execution_mode"] == "emergency"


# ---------------------------------------------------------------------------
# safe_node — agent_history written on failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_safe_node_failure_writes_agent_to_history():
    """Failed agent still appears in agent_history for tracing."""
    wrapped = safe_node(_raising_node, "geo_agent")
    result = await wrapped(_base_state())
    assert "geo_agent" in result["agent_history"]


# ---------------------------------------------------------------------------
# geo_agent — ORS failure → places returned without coordinates
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_geo_agent_ors_failure_returns_places_without_coords():
    """ORS geocoding failure → geo_agent returns places without lat/lon, no crash."""
    from agents.geo.agent import geo_agent_node

    state = {
        "request_id": "geo-fallback-test",
        "user_query": "Сванети 2 дня",
        "execution_mode": "normal",
        "agent_history": [],
        "errors": [],
        "search_context": {"region": "Svaneti"},
        "search_results": [
            {
                "name": "Ushguli",
                "location": "Svaneti, Georgia",
                "category": "village",
                "score": 0.9,
                "description": "Highest village",
                "tags": [],
                "metadata": {},  # no payload coords
            }
        ],
        "budget_state": {},
    }

    # cached_geocode_city is imported inside _geo_enrich_places at call time,
    # so we patch the source module, not the agent module.
    with patch(
        "src.tools.tool_cache.cached_geocode_city",
        side_effect=ConnectionError("ORS down"),
    ):
        with patch(
            "src.tools.tool_cache.cached_get_route",
            side_effect=ConnectionError("ORS down"),
        ):
            result = await geo_agent_node(state)

    # Must not crash — returns enriched_places without coords
    assert "enriched_places" in result
    assert isinstance(result["enriched_places"], list)
    # Place is returned even without coordinates
    assert len(result["enriched_places"]) == 1
    assert "lat" not in result["enriched_places"][0]


# ---------------------------------------------------------------------------
# orchestrator — Anthropic failure → _fallback_classification
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_anthropic_failure_uses_fallback():
    """Anthropic API failure → orchestrator returns fallback classification, no crash."""
    from orchestrator.orchestrator import OrchestratorAgent

    agent = OrchestratorAgent.__new__(OrchestratorAgent)
    agent.system_prompt = "test"

    state = {
        "request_id": "orch-fallback-test",
        "user_query": "Батуми 3 дня природа",
        "user_language": "ru",
        "has_current_plan": False,
    }

    with patch.object(
        agent,
        "_fallback_classification",
        wraps=agent._fallback_classification,
    ) as mock_fallback:
        with patch(
            "langchain_anthropic.ChatAnthropic.ainvoke",
            new_callable=AsyncMock,
            side_effect=ConnectionError("Anthropic unreachable"),
        ):
            # Need llm attribute for classify to work
            from langchain_anthropic import ChatAnthropic
            from config.settings import get_settings
            settings = get_settings()
            agent.llm = ChatAnthropic(
                model="claude-haiku-4-5-20251001",
                api_key=settings.anthropic_api_key or "test-key",
                max_tokens=100,
                temperature=0.0,
                timeout=30,
            )
            result = await agent.classify(state)

    mock_fallback.assert_called_once()
    assert result.intent in {"PLAN", "INFO", "CONSULT", "REVISE"}


# ---------------------------------------------------------------------------
# AGENT_TIMEOUTS — all agents have a timeout defined
# ---------------------------------------------------------------------------

def test_all_registered_agents_have_timeout():
    """Every agent in registry must have a timeout in AGENT_TIMEOUTS."""
    expected = {
        "search_agent",
        "geo_agent",
        "planning_agent",
        "validation_agent",
        "response_agent",
        "consultation_agent",
        "revision_agent",
    }
    missing = expected - set(AGENT_TIMEOUTS.keys())
    assert not missing, f"Missing timeouts for: {missing}"
