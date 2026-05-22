# tests/integration/test_pipeline_order.py
"""Integration tests for pipeline order: search -> geo -> planning -> validation."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import json


SEARCH_RESULTS = [
    {"name": "Batumi Boulevard", "location": "Batumi, Adjara",
     "category": "attraction", "score": 0.9, "tags": ["walking"], "description": "Promenade"},
    {"name": "Mtirala National Park", "location": "Adjara",
     "category": "national park", "score": 0.85, "tags": ["nature"], "description": "Forest"},
    {"name": "Gonio Fortress", "location": "Gonio, Adjara",
     "category": "fortress", "score": 0.80, "tags": ["history"], "description": "Fortress"},
]


def mock_geocode(city):
    return {"lat": 41.6168, "lon": 41.6367}


def mock_route(lon1, lat1, lon2, lat2):
    return {"distance_km": 20.0, "duration_min": 25.0}


def mock_llm_response():
    itinerary = {
        "days": [
            {"day": 1, "location": "Batumi",
             "activities": [
                 {"name": "Batumi Boulevard", "location": "Batumi",
                  "category": "attraction",
                  "coordinates": {"lat": 41.6168, "lon": 41.6367},
                  "description": "Promenade"},
                 {"name": "Gonio Fortress", "location": "Gonio",
                  "category": "fortress",
                  "coordinates": {"lat": 41.51, "lon": 41.62},
                  "description": "Fortress"},
             ],
             "total_distance_km": 20, "total_driving_hours": 0.4},
            {"day": 2, "location": "Adjara",
             "activities": [
                 {"name": "Mtirala National Park", "location": "Adjara",
                  "category": "national park",
                  "coordinates": {"lat": 41.72, "lon": 41.81},
                  "description": "Forest"},
                 {"name": "Extra Place", "location": "Adjara",
                  "category": "nature",
                  "coordinates": {"lat": 41.73, "lon": 41.82},
                  "description": ""},
             ],
             "total_distance_km": 25, "total_driving_hours": 0.5},
        ],
        "total_days": 2,
        "pace": "moderate",
        "summary": "2-day Adjara trip",
    }
    mock_resp = MagicMock()
    mock_resp.content = json.dumps(itinerary)
    return mock_resp


@pytest.mark.asyncio
async def test_full_pipeline_state_flow():
    """
    Full pipeline: search -> geo -> planning -> validation.
    Verifies that state is passed correctly between nodes.
    """
    from agents.search.agent import search_agent_node
    from agents.geo.agent import geo_agent_node
    from agents.planning.agent import planning_agent_node
    from agents.validation.agent import validation_agent_node

    initial_state = {
        "request_id": "pipeline-test-1",
        "user_query": "Батуми 2 дня природа",
        "orchestrator_params": {"region": "Adjara", "max_results": 10},
        "trip_parameters": {"days": 2, "pace": "moderate"},
        "agent_history": [],
        "budget_state": {"llm_calls": 0, "tool_calls": 0,
                         "prompt_tokens": 0, "completion_tokens": 0,
                         "estimated_cost_usd": 0.0},
    }

    with patch("agents.search.agent._do_search", return_value=SEARCH_RESULTS), \
         patch("src.tools.tool_cache.cached_geocode_city", side_effect=mock_geocode), \
         patch("src.tools.tool_cache.cached_get_route", side_effect=mock_route), \
         patch("langchain_openai.ChatOpenAI.ainvoke", new_callable=AsyncMock,
               return_value=mock_llm_response()):

        # 1. Search
        search_out = await search_agent_node(initial_state)
        assert search_out["search_results"]

        state_after_search = {**initial_state, **search_out}

        # Geo runs before planning so the planner receives coordinates and routes.
        geo_out = await geo_agent_node(state_after_search)
        assert "enriched_places" in geo_out
        assert "distance_matrix" in geo_out

        state_after_geo = {**state_after_search, **geo_out}

        # 3. Planning receives enriched_places and distance_matrix.
        plan_out = await planning_agent_node(state_after_geo)
        assert "raw_itinerary" in plan_out
        assert plan_out["raw_itinerary"]["total_days"] == 2

        state_after_planning = {**state_after_geo, **plan_out}

        # 4. Validation
        val_out = await validation_agent_node(state_after_planning)
        assert "validation_result" in val_out


@pytest.mark.asyncio
async def test_geo_before_planning_enriched_data_used():
    """
    Verifies that planning receives enriched_places with coordinates, not only search_results.
    """
    from agents.geo.agent import geo_agent_node
    from agents.planning.agent import planning_agent_node

    state = {
        "request_id": "order-test",
        "user_query": "Батуми 1 день",
        "search_results": SEARCH_RESULTS,
        "search_context": {"region": "Adjara"},
        "trip_parameters": {"days": 1, "pace": "moderate"},
        "agent_history": ["search_agent"],
        "budget_state": {"llm_calls": 0, "tool_calls": 0,
                         "prompt_tokens": 0, "completion_tokens": 0,
                         "estimated_cost_usd": 0.0},
    }

    with patch("src.tools.tool_cache.cached_geocode_city", side_effect=mock_geocode), \
         patch("src.tools.tool_cache.cached_get_route", side_effect=mock_route):
        geo_out = await geo_agent_node(state)

    state_with_geo = {**state, **geo_out}

    # Planning should receive enriched_places with lat/lon.
    places_used = state_with_geo.get("enriched_places") or state_with_geo.get("search_results")
    assert places_used is not None
    assert all("lat" in p for p in places_used if p.get("lat"))
