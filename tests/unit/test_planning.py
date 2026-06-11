# tests/unit/test_planning.py
"""Unit tests for agents/planning/agent.py."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agents.planning.agent import (
    _format_rich_places,
    _group_by_proximity,
    _simple_split,
    _try_parse_json,
    _parse_candidate,
)


class TestFormatRichPlaces:
    def test_includes_coordinates(self, places_geocoded):
        text = _format_rich_places(places_geocoded, {})
        assert "lat=41" in text
        assert "lon=41" in text

    def test_includes_tags(self, places_geocoded):
        text = _format_rich_places(places_geocoded, {})
        assert "hiking" in text or "nature" in text

    def test_includes_distances(self, places_geocoded, distance_matrix):
        text = _format_rich_places(places_geocoded, distance_matrix)
        assert "DISTANCES" in text
        assert "km" in text

    def test_unknown_coords_when_no_lat(self):
        places = [{"name": "Place", "location": "Tbilisi", "category": "museum",
                   "tags": [], "description": ""}]
        text = _format_rich_places(places, {})
        assert "coordinates: unknown" in text

    def test_max_20_places(self, places_geocoded):
        many = places_geocoded * 10  # 30 places
        text = _format_rich_places(many, {})
        # There should be no more than 20 [N] entries.
        count = text.count("\n[")
        assert count <= 20


class TestGroupByProximity:
    def test_groups_nearest_together(self, places_geocoded, distance_matrix):
        result = _group_by_proximity(places_geocoded, distance_matrix, days=2, pace="moderate")
        assert len(result) == 2
        # Each day has activities.
        for day in result:
            assert len(day["activities"]) >= 1

    def test_correct_day_numbers(self, places_geocoded, distance_matrix):
        result = _group_by_proximity(places_geocoded, distance_matrix, days=2, pace="moderate")
        assert result[0]["day"] == 1
        assert result[1]["day"] == 2

    def test_activities_have_coordinates(self, places_geocoded, distance_matrix):
        result = _group_by_proximity(places_geocoded, distance_matrix, days=2, pace="moderate")
        for day in result:
            for act in day["activities"]:
                assert "coordinates" in act

    def test_empty_places_returns_fallback(self):
        result = _group_by_proximity([], {}, days=2, pace="moderate")
        assert result == [{"day": 1, "location": "Georgia", "activities": []}]

    def test_relaxed_pace_fewer_per_day(self, places_geocoded, distance_matrix):
        result = _group_by_proximity(places_geocoded, distance_matrix, days=3, pace="relaxed")
        # relaxed = 2 per day max
        for day in result:
            assert len(day["activities"]) <= 2


class TestSimpleSplit:
    def test_splits_into_days(self, places_geocoded):
        result = _simple_split(places_geocoded, days=2, pace="moderate", region="Adjara")
        assert len(result) == 2

    def test_moderate_3_per_day(self):
        places = [{"name": f"Place{i}", "location": "Tbilisi", "category": "x",
                   "description": "", "tags": []} for i in range(9)]
        result = _simple_split(places, days=3, pace="moderate", region="Tbilisi")
        assert len(result) == 3
        assert len(result[0]["activities"]) == 3


class TestTryParseJson:
    def test_plain_json(self):
        text = '{"days": [{"day": 1, "activities": []}], "total_days": 1}'
        result = _try_parse_json(text, "moderate", 1)
        assert result is not None
        assert result["total_days"] == 1

    def test_json_in_markdown_block(self):
        text = '```json\n{"days": [{"day": 1, "activities": []}]}\n```'
        result = _try_parse_json(text, "moderate", 1)
        assert result is not None

    def test_json_with_text_around(self):
        text = 'Here is your itinerary:\n{"days": [{"day": 1, "activities": []}]}\nEnjoy!'
        result = _try_parse_json(text, "moderate", 1)
        assert result is not None

    def test_invalid_json_returns_none(self):
        assert _try_parse_json("not json at all", "moderate", 3) is None

    def test_defaults_added(self):
        text = '{"days": [{"day": 1, "activities": []}]}'
        result = _try_parse_json(text, "relaxed", 1)
        assert result["pace"] == "relaxed"
        assert "summary" in result


# ---------------------------------------------------------------------------
# planning_agent_node — region resolution (Debt AK)
# ---------------------------------------------------------------------------

def _make_places():
    return [
        {"name": "Ushguli", "location": "Svaneti", "category": "village",
         "tags": ["nature"], "description": "Remote village",
         "metadata": {"lat": 43.1, "lon": 42.9}},
    ]


@pytest.mark.asyncio
async def test_planning_agent_uses_trip_parameters_region_when_search_context_absent():
    """On REVISE, search_context is absent; region must come from trip_parameters."""
    captured_prompts: list[str] = []

    class _FakeLLM:
        async def ainvoke(self, prompt, config=None):
            captured_prompts.append(str(prompt))
            return MagicMock(content='{"days":[{"day":1,"location":"Svaneti",'
                             '"activities":[{"name":"Hike","duration_hours":2,'
                             '"distance_km":5}],"total_distance_km":5,'
                             '"total_driving_hours":0}],'
                             '"total_days":1,"pace":"moderate","summary":"ok"}')

    class _FakeTracker:
        def __init__(self, **kw): pass

    # ChatOpenAI, TokenTracker, merge_budget are lazy-imported inside the function —
    # patch at their source modules, not at agents.planning.agent.
    with patch("langchain_openai.ChatOpenAI", return_value=_FakeLLM()), \
         patch("monitoring.token_tracker.TokenTracker", _FakeTracker), \
         patch("monitoring.token_tracker.merge_budget", lambda b, t: b):
        from agents.planning.agent import planning_agent_node
        await planning_agent_node({
            "request_id": "ak-test",
            "user_query": "revise my Svaneti trip",
            "user_language": "en",
            "enriched_places": _make_places(),
            "search_results": [],
            "distance_matrix": {},
            "trip_parameters": {"region": "Svaneti", "days": 1, "pace": "moderate"},
            "agent_history": [],
            "errors": [],
            "budget_state": {},
        })

    assert captured_prompts, "LLM must be called"
    assert "Svaneti" in captured_prompts[0], (
        "Region from trip_parameters must appear in the prompt when search_context is absent"
    )


@pytest.mark.asyncio
async def test_planning_agent_search_context_region_takes_priority():
    """search_context.region overrides trip_parameters.region when both are present."""
    captured_prompts: list[str] = []

    class _FakeLLM:
        async def ainvoke(self, prompt, config=None):
            captured_prompts.append(str(prompt))
            return MagicMock(content='{"days":[{"day":1,"location":"Kakheti",'
                             '"activities":[{"name":"Wine tour","duration_hours":3,'
                             '"distance_km":10}],"total_distance_km":10,'
                             '"total_driving_hours":0}],'
                             '"total_days":1,"pace":"moderate","summary":"ok"}')

    class _FakeTracker:
        def __init__(self, **kw): pass

    with patch("langchain_openai.ChatOpenAI", return_value=_FakeLLM()), \
         patch("monitoring.token_tracker.TokenTracker", _FakeTracker), \
         patch("monitoring.token_tracker.merge_budget", lambda b, t: b):
        from agents.planning.agent import planning_agent_node
        await planning_agent_node({
            "request_id": "ak-test-2",
            "user_query": "plan my Kakheti trip",
            "user_language": "en",
            "enriched_places": _make_places(),
            "search_results": [],
            "distance_matrix": {},
            "search_context": {"region": "Kakheti"},
            "trip_parameters": {"region": "Tbilisi", "days": 1, "pace": "moderate"},
            "agent_history": [],
            "errors": [],
            "budget_state": {},
        })

    assert "Kakheti" in captured_prompts[0]
