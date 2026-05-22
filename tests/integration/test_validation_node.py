# tests/integration/test_validation_node.py
"""Integration tests for validation_agent_node without LLM calls."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock


@pytest.fixture
def state_with_itinerary(valid_itinerary, distance_matrix):
    return {
        "request_id": "val-test-1",
        "raw_itinerary": valid_itinerary,
        "distance_matrix": distance_matrix,
        "trip_parameters": {"pace": "moderate"},
        "agent_history": ["search_agent", "geo_agent", "planning_agent"],
        "budget_state": {"llm_calls": 0, "tool_calls": 0,
                         "prompt_tokens": 0, "completion_tokens": 0,
                         "estimated_cost_usd": 0.0},
    }


@pytest.mark.asyncio
async def test_valid_itinerary_no_llm_call(state_with_itinerary):
    """A clean itinerary passes without an LLM call."""
    with patch("langchain_openai.ChatOpenAI.ainvoke") as mock_llm:
        from agents.validation.agent import validation_agent_node
        result = await validation_agent_node(state_with_itinerary)

    mock_llm.assert_not_called()
    assert result["validation_result"]["is_valid"] is True


@pytest.mark.asyncio
async def test_duplicate_places_caught(state_with_itinerary):
    """A repeated place across days is treated as an error."""
    itinerary = {
        "days": [
            {"day": 1, "activities": [{"name": "Batumi Boulevard"}, {"name": "Gonio"}],
             "total_distance_km": 15, "total_driving_hours": 0.3},
            {"day": 2, "activities": [{"name": "Batumi Boulevard"}, {"name": "Mtirala"}],
             "total_distance_km": 25, "total_driving_hours": 0.5},
        ]
    }
    state_with_itinerary["raw_itinerary"] = itinerary
    from agents.validation.agent import validation_agent_node
    result = await validation_agent_node(state_with_itinerary)
    assert result["validation_result"]["is_valid"] is False
    assert any("already appeared" in e for e in result["validation_result"]["errors"])


@pytest.mark.asyncio
async def test_single_activity_day_warning(state_with_itinerary):
    """A one-activity day is treated as a warning."""
    itinerary = {
        "days": [
            {"day": 1, "activities": [{"name": "Solo Place"}],
             "total_distance_km": 0, "total_driving_hours": 0},
        ]
    }
    state_with_itinerary["raw_itinerary"] = itinerary
    from agents.validation.agent import validation_agent_node
    result = await validation_agent_node(state_with_itinerary)
    assert result["validation_result"]["is_valid"] is True
    assert result["validation_result"]["warnings"]


@pytest.mark.asyncio
async def test_no_itinerary_returns_abort():
    from agents.validation.agent import validation_agent_node
    state = {"request_id": "val-empty", "agent_history": []}
    result = await validation_agent_node(state)
    assert result["validation_result"]["recommended_action"] == "abort"
    assert result["validation_result"]["is_valid"] is False


@pytest.mark.asyncio
async def test_exceeds_km_triggers_retry(state_with_itinerary):
    """Exceeding the distance limit sets recommended_action to retry."""
    itinerary = {
        "days": [
            {"day": 1, "activities": [{"name": "A"}, {"name": "B"}],
             "total_distance_km": 300, "total_driving_hours": 1},
        ]
    }
    state_with_itinerary["raw_itinerary"] = itinerary
    from agents.validation.agent import validation_agent_node
    result = await validation_agent_node(state_with_itinerary)
    assert result["validation_result"]["recommended_action"] == "retry"


@pytest.mark.asyncio
async def test_after_two_planning_attempts_no_retry(state_with_itinerary):
    """After two planning attempts, avoid retry loops."""
    state_with_itinerary["agent_history"] = [
        "planning_agent", "validation_agent", "planning_agent"
    ]
    itinerary = {
        "days": [
            {"day": 1, "activities": [{"name": "A"}, {"name": "B"}],
             "total_distance_km": 300, "total_driving_hours": 1},
        ]
    }
    state_with_itinerary["raw_itinerary"] = itinerary
    from agents.validation.agent import validation_agent_node
    result = await validation_agent_node(state_with_itinerary)
    assert result["validation_result"]["recommended_action"] == "proceed"
