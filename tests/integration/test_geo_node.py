# tests/integration/test_geo_node.py
"""Integration tests for geo_agent_node with a mocked ORS API."""

import pytest
from unittest.mock import patch, MagicMock


def mock_geocode(city: str) -> dict:
    coords = {
        "Batumi":   {"lat": 41.6168, "lon": 41.6367},
        "Adjara":   {"lat": 41.7200, "lon": 41.8100},
        "Gonio":    {"lat": 41.5100, "lon": 41.6200},
        "Mtirala":  {"lat": 41.7200, "lon": 41.8100},
    }
    for key, val in coords.items():
        if key.lower() in city.lower():
            return val
    return {"error": f"not found: {city}"}


def mock_route(lon1, lat1, lon2, lat2) -> dict:
    return {"distance_km": 25.0, "duration_min": 30.0}


@pytest.fixture
def state_with_search(places_geocoded):
    return {
        "request_id": "geo-test-1",
        "search_results": [
            {"name": "Batumi Boulevard", "location": "Batumi, Adjara",
             "category": "attraction", "tags": [], "description": ""},
            {"name": "Mtirala National Park", "location": "Adjara",
             "category": "national park", "tags": ["nature"], "description": ""},
            {"name": "Gonio Fortress", "location": "Gonio, Adjara",
             "category": "fortress", "tags": ["history"], "description": ""},
        ],
    }


@pytest.mark.asyncio
async def test_geo_enriches_with_lat_lon(state_with_search):
    """geo_agent_node adds lat/lon to places."""
    with patch("src.tools.tool_cache.cached_geocode_city", side_effect=mock_geocode), \
         patch("src.tools.tool_cache.cached_get_route", side_effect=mock_route):
        from agents.geo.agent import geo_agent_node
        result = await geo_agent_node(state_with_search)

    assert "enriched_places" in result
    geocoded = [p for p in result["enriched_places"] if p.get("lat")]
    assert len(geocoded) > 0


@pytest.mark.asyncio
async def test_geo_builds_distance_matrix(state_with_search):
    """geo_agent_node builds distance_matrix."""
    with patch("src.tools.tool_cache.cached_geocode_city", side_effect=mock_geocode), \
         patch("src.tools.tool_cache.cached_get_route", side_effect=mock_route):
        from agents.geo.agent import geo_agent_node
        result = await geo_agent_node(state_with_search)

    assert "distance_matrix" in result
    assert len(result["distance_matrix"]) > 0


@pytest.mark.asyncio
async def test_geo_reverse_routes_added(state_with_search):
    """Reverse routes are added to distance_matrix."""
    with patch("src.tools.tool_cache.cached_geocode_city", side_effect=mock_geocode), \
         patch("src.tools.tool_cache.cached_get_route", side_effect=mock_route):
        from agents.geo.agent import geo_agent_node
        result = await geo_agent_node(state_with_search)

    dm = result.get("distance_matrix", {})
    for key in list(dm.keys()):
        parts = key.split("→")
        if len(parts) == 2:
            reverse = f"{parts[1]}→{parts[0]}"
            assert reverse in dm, f"Reverse key '{reverse}' missing for '{key}'"


@pytest.mark.asyncio
async def test_geo_no_search_results_returns_error():
    """geo_agent returns an error when search_results is empty."""
    from agents.geo.agent import geo_agent_node
    state = {"request_id": "geo-test-empty", "search_results": []}
    result = await geo_agent_node(state)
    assert "errors" in result
    assert "geo_agent" in result["agent_history"]


@pytest.mark.asyncio
async def test_geo_no_llm_calls(state_with_search):
    """geo_agent_node does not make LLM calls."""
    with patch("src.tools.tool_cache.cached_geocode_city", side_effect=mock_geocode), \
         patch("src.tools.tool_cache.cached_get_route", side_effect=mock_route):
        from agents.geo.agent import geo_agent_node
        result = await geo_agent_node(state_with_search)

    # budget_state is absent or llm_calls == 0.
    budget = result.get("budget_state", {})
    assert budget.get("llm_calls", 0) == 0


@pytest.mark.asyncio
async def test_geo_partial_geocoding(state_with_search):
    """If some places fail geocoding, all places remain in enriched_places."""
    def partial_geocode(city):
        if "Batumi" in city:
            return {"lat": 41.6168, "lon": 41.6367}
        return {"error": "not found"}

    with patch("src.tools.tool_cache.cached_geocode_city", side_effect=partial_geocode), \
         patch("src.tools.tool_cache.cached_get_route", side_effect=mock_route):
        from agents.geo.agent import geo_agent_node
        result = await geo_agent_node(state_with_search)

    # All places remain in enriched_places, even without coordinates.
    assert len(result["enriched_places"]) == len(state_with_search["search_results"])
