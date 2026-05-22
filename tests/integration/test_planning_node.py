# tests/integration/test_planning_node.py
"""Integration tests for planning_agent_node with a mocked LLM."""

import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock


def _make_llm_response(days: int = 2, pace: str = "moderate") -> MagicMock:
    """Build a mocked LLM response with valid itinerary JSON."""
    itinerary = {
        "days": [
            {
                "day": i + 1,
                "location": "Batumi",
                "activities": [
                    {"name": f"Place{i*2+1}", "location": "Batumi",
                     "category": "attraction",
                     "coordinates": {"lat": 41.6, "lon": 41.6},
                     "description": "Test place"},
                    {"name": f"Place{i*2+2}", "location": "Adjara",
                     "category": "nature",
                     "coordinates": {"lat": 41.7, "lon": 41.8},
                     "description": "Test nature"},
                ],
                "total_distance_km": 25,
                "total_driving_hours": 0.5,
            }
            for i in range(days)
        ],
        "total_days": days,
        "pace": pace,
        "summary": f"Test {days}-day trip",
    }
    mock_response = MagicMock()
    mock_response.content = json.dumps(itinerary)
    return mock_response


@pytest.fixture
def state_with_geo(places_geocoded, distance_matrix):
    return {
        "request_id": "plan-test-1",
        "user_query": "Батуми 2 дня природа",
        "enriched_places": places_geocoded,
        "distance_matrix": distance_matrix,
        "search_context": {"region": "Adjara"},
        "trip_parameters": {"days": 2, "pace": "moderate"},
        "agent_history": [],
        "budget_state": {"llm_calls": 0, "tool_calls": 0,
                         "prompt_tokens": 0, "completion_tokens": 0,
                         "estimated_cost_usd": 0.0},
    }


@pytest.mark.asyncio
async def test_planning_uses_enriched_places(state_with_geo):
    """planning_agent reads enriched_places from GeoAgent."""
    mock_resp = _make_llm_response(days=2)
    with patch("langchain_openai.ChatOpenAI.ainvoke", new_callable=AsyncMock,
               return_value=mock_resp):
        from agents.planning.agent import planning_agent_node
        result = await planning_agent_node(state_with_geo)

    assert "raw_itinerary" in result
    assert result["raw_itinerary"]["total_days"] == 2


@pytest.mark.asyncio
async def test_planning_one_llm_call(state_with_geo):
    """planning_agent makes exactly one LLM call."""
    mock_resp = _make_llm_response(days=2)
    call_count = {"n": 0}

    async def counting_invoke(*args, **kwargs):
        call_count["n"] += 1
        return mock_resp

    with patch("langchain_openai.ChatOpenAI.ainvoke", side_effect=counting_invoke):
        from agents.planning.agent import planning_agent_node
        await planning_agent_node(state_with_geo)

    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_planning_fallback_on_bad_json(state_with_geo):
    """Fallback is used when the LLM returns invalid JSON."""
    mock_resp = MagicMock()
    mock_resp.content = "Sorry, I cannot create a plan."
    with patch("langchain_openai.ChatOpenAI.ainvoke", new_callable=AsyncMock,
               return_value=mock_resp):
        from agents.planning.agent import planning_agent_node
        result = await planning_agent_node(state_with_geo)

    assert "raw_itinerary" in result
    assert result["raw_itinerary"]["days"]  # fallback should return something usable


@pytest.mark.asyncio
async def test_planning_no_places_returns_error():
    from agents.planning.agent import planning_agent_node
    state = {
        "request_id": "plan-empty",
        "user_query": "test",
        "enriched_places": [],
        "search_results": [],
        "agent_history": [],
        "budget_state": {},
    }
    result = await planning_agent_node(state)
    assert "errors" in result
    assert "planning_agent" in result["agent_history"]


@pytest.mark.asyncio
async def test_planning_fallback_uses_search_results_when_no_enriched():
    """When enriched_places is missing, the planner uses search_results."""
    mock_resp = _make_llm_response(days=1)
    state = {
        "request_id": "plan-fallback",
        "user_query": "Тбилиси 1 день",
        "enriched_places": None,  # no geo data
        "search_results": [
            {"name": "Narikala Fortress", "location": "Tbilisi",
             "category": "fortress", "tags": [], "description": ""},
            {"name": "Old Town", "location": "Tbilisi",
             "category": "district", "tags": [], "description": ""},
        ],
        "search_context": {"region": "Tbilisi"},
        "trip_parameters": {"days": 1, "pace": "moderate"},
        "agent_history": [],
        "budget_state": {"llm_calls": 0, "tool_calls": 0,
                         "prompt_tokens": 0, "completion_tokens": 0,
                         "estimated_cost_usd": 0.0},
    }
    with patch("langchain_openai.ChatOpenAI.ainvoke", new_callable=AsyncMock,
               return_value=mock_resp):
        from agents.planning.agent import planning_agent_node
        result = await planning_agent_node(state)

    assert "raw_itinerary" in result
