# tests/unit/test_planning.py
"""Unit tests for agents/planning/agent.py."""

import pytest
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
