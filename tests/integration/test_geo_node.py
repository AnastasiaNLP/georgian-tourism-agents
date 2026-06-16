# tests/integration/test_geo_node.py
"""Integration tests for geo_agent_node with a mocked ORS API."""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock


def mock_geocode(city: str, focus_lat: float = 41.9, focus_lon: float = 44.0) -> dict:
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
    with patch("src.tools.tool_cache.cached_geocode_city", new_callable=AsyncMock, side_effect=mock_geocode), \
         patch("src.tools.tool_cache.cached_get_route", new_callable=AsyncMock, side_effect=mock_route):
        from agents.geo.agent import geo_agent_node
        result = await geo_agent_node(state_with_search)

    assert "enriched_places" in result
    geocoded = [p for p in result["enriched_places"] if p.get("lat")]
    assert len(geocoded) > 0


@pytest.mark.asyncio
async def test_geo_builds_distance_matrix(state_with_search):
    """geo_agent_node builds distance_matrix."""
    with patch("src.tools.tool_cache.cached_geocode_city", new_callable=AsyncMock, side_effect=mock_geocode), \
         patch("src.tools.tool_cache.cached_get_route", new_callable=AsyncMock, side_effect=mock_route):
        from agents.geo.agent import geo_agent_node
        result = await geo_agent_node(state_with_search)

    assert "distance_matrix" in result
    assert len(result["distance_matrix"]) > 0


@pytest.mark.asyncio
async def test_geo_reverse_routes_added(state_with_search):
    """Reverse routes are added to distance_matrix."""
    with patch("src.tools.tool_cache.cached_geocode_city", new_callable=AsyncMock, side_effect=mock_geocode), \
         patch("src.tools.tool_cache.cached_get_route", new_callable=AsyncMock, side_effect=mock_route):
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
    with patch("src.tools.tool_cache.cached_geocode_city", new_callable=AsyncMock, side_effect=mock_geocode), \
         patch("src.tools.tool_cache.cached_get_route", new_callable=AsyncMock, side_effect=mock_route):
        from agents.geo.agent import geo_agent_node
        result = await geo_agent_node(state_with_search)

    # budget_state is absent or llm_calls == 0.
    budget = result.get("budget_state", {})
    assert budget.get("llm_calls", 0) == 0


@pytest.mark.asyncio
async def test_geo_uses_payload_centroid_as_ors_focus_point():
    """
    When places have Qdrant payload coordinates, geo_agent must pass their centroid
    as focus_lat/focus_lon to cached_geocode_city, not the Georgia centroid (41.9, 44.0).
    """
    # Two payload-coords places in Svaneti area (~43.0N, 42.9E)
    state = {
        "request_id": "focus-test",
        "search_results": [
            {
                "name": "Ushguli Towers",
                "location": "Svaneti",
                "category": "historic",
                "tags": [],
                "description": "",
                "metadata": {"lat": 43.12, "lon": 42.88},
            },
            {
                "name": "Mestia Cathedral",
                "location": "Svaneti",
                "category": "church",
                "tags": [],
                "description": "",
                "metadata": {"lat": 43.05, "lon": 42.73},
            },
            {
                "name": "Svaneti Waterfall",  # no payload coords — should use ORS
                "location": "Svaneti",
                "category": "waterfall",
                "tags": [],
                "description": "",
                "metadata": {},
            },
        ],
        "search_context": {"region": "Svaneti"},
    }

    captured_focus: list[tuple] = []

    async def _mock_geocode(city: str, focus_lat: float = 41.9, focus_lon: float = 44.0) -> dict:
        captured_focus.append((focus_lat, focus_lon))
        return {"lat": 43.05, "lon": 42.73}

    with patch("src.tools.tool_cache.cached_geocode_city", side_effect=_mock_geocode), \
         patch("src.tools.tool_cache.cached_get_route", new_callable=AsyncMock,
               return_value={"distance_km": 10.0, "duration_min": 15.0}):
        from agents.geo.agent import geo_agent_node
        await geo_agent_node(state)

    # ORS was called exactly once (for the waterfall — others used payload coords)
    assert len(captured_focus) == 1, f"Expected 1 ORS call, got {len(captured_focus)}"

    used_lat, used_lon = captured_focus[0]
    # Expected centroid of Ushguli (43.12, 42.88) and Mestia (43.05, 42.73)
    expected_lat = (43.12 + 43.05) / 2  # ≈ 43.085
    expected_lon = (42.88 + 42.73) / 2  # ≈ 42.805
    assert abs(used_lat - expected_lat) < 0.01, f"focus_lat {used_lat:.3f} != centroid {expected_lat:.3f}"
    assert abs(used_lon - expected_lon) < 0.01, f"focus_lon {used_lon:.3f} != centroid {expected_lon:.3f}"

    # Centroid must not be the Georgia default
    assert used_lat != 41.9 and used_lon != 44.0, "Must not fall back to Georgia centroid"


@pytest.mark.asyncio
async def test_geo_falls_back_to_georgia_centroid_when_no_payload_coords():
    """When no places have payload coordinates, geo_agent falls back to Georgia centroid."""
    state = {
        "request_id": "focus-fallback-test",
        "search_results": [
            {
                "name": "Mystery Waterfall",
                "location": "Georgia",
                "category": "waterfall",
                "tags": [],
                "description": "",
                "metadata": {},  # no payload coords
            },
        ],
    }

    captured_focus: list[tuple] = []

    async def _mock_geocode(city: str, focus_lat: float = 41.9, focus_lon: float = 44.0) -> dict:
        captured_focus.append((focus_lat, focus_lon))
        return {"lat": 42.0, "lon": 43.5}

    with patch("src.tools.tool_cache.cached_geocode_city", side_effect=_mock_geocode), \
         patch("src.tools.tool_cache.cached_get_route", new_callable=AsyncMock,
               return_value={"distance_km": 0.0, "duration_min": 0.0}):
        from agents.geo.agent import geo_agent_node
        await geo_agent_node(state)

    assert len(captured_focus) == 1
    used_lat, used_lon = captured_focus[0]
    assert used_lat == 41.9 and used_lon == 44.0, "Must use Georgia centroid when no payload coords"


def _svaneti_state_with_ors_place():
    """Two payload-coords places in Svaneti + one ORS-only place."""
    return {
        "request_id": "envelope-test",
        "search_results": [
            {"name": "Ushguli Towers", "location": "Svaneti", "category": "historic",
             "tags": [], "description": "", "metadata": {"lat": 43.12, "lon": 42.88}},
            {"name": "Mestia Cathedral", "location": "Svaneti", "category": "church",
             "tags": [], "description": "", "metadata": {"lat": 43.05, "lon": 42.73}},
            {"name": "Svaneti Waterfall", "location": "Svaneti", "category": "waterfall",
             "tags": [], "description": "", "metadata": {}},
        ],
    }


@pytest.mark.asyncio
async def test_geo_discards_ors_coord_outside_region_envelope():
    """An ORS result inside Georgia but far from the payload cluster is discarded."""
    state = _svaneti_state_with_ors_place()

    async def _mock_geocode(city, focus_lat=41.9, focus_lon=44.0):
        # Tbilisi — inside Georgia, but ~250 km from the Svaneti payload cluster.
        return {"lat": 41.72, "lon": 44.79}

    with patch("src.tools.tool_cache.cached_geocode_city", side_effect=_mock_geocode), \
         patch("src.tools.tool_cache.cached_get_route", new_callable=AsyncMock,
               return_value={"distance_km": 10.0, "duration_min": 15.0}):
        from agents.geo.agent import geo_agent_node
        result = await geo_agent_node(state)

    by_name = {p["name"]: p for p in result["enriched_places"]}
    # Payload places keep their coordinates.
    assert by_name["Ushguli Towers"].get("lat") == 43.12
    assert by_name["Mestia Cathedral"].get("lat") == 43.05
    # The far ORS result is rejected — the place has no coordinates.
    assert by_name["Svaneti Waterfall"].get("lat") is None


@pytest.mark.asyncio
async def test_geo_keeps_ors_coord_inside_region_envelope():
    """An ORS result within the payload cluster envelope is accepted."""
    state = _svaneti_state_with_ors_place()

    async def _mock_geocode(city, focus_lat=41.9, focus_lon=44.0):
        # Inside the Svaneti envelope.
        return {"lat": 43.00, "lon": 42.80}

    with patch("src.tools.tool_cache.cached_geocode_city", side_effect=_mock_geocode), \
         patch("src.tools.tool_cache.cached_get_route", new_callable=AsyncMock,
               return_value={"distance_km": 10.0, "duration_min": 15.0}):
        from agents.geo.agent import geo_agent_node
        result = await geo_agent_node(state)

    by_name = {p["name"]: p for p in result["enriched_places"]}
    assert by_name["Svaneti Waterfall"].get("lat") == 43.00
    assert by_name["Svaneti Waterfall"].get("coord_source") == "ors"


@pytest.mark.asyncio
async def test_geo_single_anchor_does_not_build_envelope():
    """With only one payload anchor, the envelope is not applied (country box only)."""
    state = {
        "request_id": "single-anchor-test",
        "search_results": [
            {"name": "Lone Anchor", "location": "Kakheti", "category": "winery",
             "tags": [], "description": "", "metadata": {"lat": 41.92, "lon": 45.49}},
            {"name": "Far Same-Region Place", "location": "Kakheti", "category": "monastery",
             "tags": [], "description": "", "metadata": {}},
        ],
    }

    async def _mock_geocode(city, focus_lat=41.9, focus_lon=44.0):
        # ~120 km from the lone anchor but legitimately inside Georgia.
        return {"lat": 41.30, "lon": 46.30}

    with patch("src.tools.tool_cache.cached_geocode_city", side_effect=_mock_geocode), \
         patch("src.tools.tool_cache.cached_get_route", new_callable=AsyncMock,
               return_value={"distance_km": 10.0, "duration_min": 15.0}):
        from agents.geo.agent import geo_agent_node
        result = await geo_agent_node(state)

    by_name = {p["name"]: p for p in result["enriched_places"]}
    # No envelope with a single anchor → the far ORS result is kept.
    assert by_name["Far Same-Region Place"].get("lat") == 41.30


@pytest.mark.asyncio
async def test_geo_bounds_tool_concurrency():
    """Concurrent geocode calls never exceed the configured concurrency limit."""
    n = 15
    state = {
        "request_id": "concurrency-test",
        "search_results": [
            {"name": f"Place {i}", "location": "Georgia", "category": "x",
             "tags": [], "description": "", "metadata": {}}
            for i in range(n)
        ],
    }
    tracker = {"cur": 0, "peak": 0}

    async def _mock_geocode(city, focus_lat=41.9, focus_lon=44.0):
        tracker["cur"] += 1
        tracker["peak"] = max(tracker["peak"], tracker["cur"])
        await asyncio.sleep(0.01)
        tracker["cur"] -= 1
        return {"lat": 42.0, "lon": 43.5}

    mock_settings = MagicMock(geo_max_concurrency=3)

    with patch("src.tools.tool_cache.cached_geocode_city", side_effect=_mock_geocode), \
         patch("src.tools.tool_cache.cached_get_route", new_callable=AsyncMock,
               return_value={"distance_km": 10.0, "duration_min": 15.0}), \
         patch("config.settings.get_settings", return_value=mock_settings):
        from agents.geo.agent import geo_agent_node
        await geo_agent_node(state)

    assert tracker["peak"] <= 3, f"peak concurrency {tracker['peak']} exceeded limit 3"
    assert tracker["peak"] > 1, "expected real concurrency, not fully serialized"


@pytest.mark.asyncio
async def test_geo_partial_geocoding(state_with_search):
    """If some places fail geocoding, all places remain in enriched_places."""
    def partial_geocode(city):
        if "Batumi" in city:
            return {"lat": 41.6168, "lon": 41.6367}
        return {"error": "not found"}

    with patch("src.tools.tool_cache.cached_geocode_city", new_callable=AsyncMock, side_effect=partial_geocode), \
         patch("src.tools.tool_cache.cached_get_route", new_callable=AsyncMock, side_effect=mock_route):
        from agents.geo.agent import geo_agent_node
        result = await geo_agent_node(state_with_search)

    # All places remain in enriched_places, even without coordinates.
    assert len(result["enriched_places"]) == len(state_with_search["search_results"])
