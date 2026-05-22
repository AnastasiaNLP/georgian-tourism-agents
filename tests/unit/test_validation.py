# tests/unit/test_validation.py
"""Unit tests for agents/validation/agent.py and _auto_validate."""

import pytest
from agents.validation.agent import _auto_validate, _normalize_place_name


class TestAutoValidate:
    def test_valid_itinerary_passes(self, valid_itinerary):
        result = _auto_validate(valid_itinerary, "moderate", {})
        assert result["is_valid"] is True
        assert result["errors"] == []

    def test_p16_single_activity_is_warning(self):
        itinerary = {
            "days": [{"day": 1, "activities": [{"name": "Place A"}],
                      "total_distance_km": 0, "total_driving_hours": 0}]
        }
        result = _auto_validate(itinerary, "moderate", {})
        assert any("only 1 activity" in w for w in result["warnings"])
        assert result["is_valid"] is True  # warning, not error

    def test_p17_duplicate_place_is_error(self):
        itinerary = {
            "days": [
                {"day": 1, "activities": [{"name": "Batumi Boulevard"},
                                           {"name": "Gonio Fortress"}],
                 "total_distance_km": 15, "total_driving_hours": 0.3},
                {"day": 2, "activities": [{"name": "Batumi Boulevard"},  # duplicate
                                           {"name": "Mtirala Park"}],
                 "total_distance_km": 25, "total_driving_hours": 0.5},
            ]
        }
        result = _auto_validate(itinerary, "moderate", {})
        assert any("already appeared" in e for e in result["errors"])
        assert result["is_valid"] is False

    def test_p11_zero_distance_with_multiple_places_is_warning(self):
        itinerary = {
            "days": [{"day": 1,
                      "activities": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
                      "total_distance_km": 0, "total_driving_hours": 0}]
        }
        result = _auto_validate(itinerary, "moderate", {})
        assert any("total_distance_km=0" in w for w in result["warnings"])

    def test_exceeds_km_limit_is_error(self):
        itinerary = {
            "days": [{"day": 1,
                      "activities": [{"name": "A"}, {"name": "B"}],
                      "total_distance_km": 200, "total_driving_hours": 3}]
        }
        result = _auto_validate(itinerary, "moderate", {})  # limit=150
        assert any("exceeds" in e for e in result["errors"])
        assert result["is_valid"] is False

    def test_exceeds_activities_limit_is_error(self):
        itinerary = {
            "days": [{"day": 1,
                      "activities": [{"name": f"Place{i}"} for i in range(7)],
                      "total_distance_km": 50, "total_driving_hours": 1}]
        }
        result = _auto_validate(itinerary, "moderate", {})  # max=4
        assert any("activities >" in e for e in result["errors"])

    def test_exceeds_driving_hours_is_error(self):
        itinerary = {
            "days": [{"day": 1,
                      "activities": [{"name": "A"}, {"name": "B"}],
                      "total_distance_km": 100, "total_driving_hours": 5}]
        }
        result = _auto_validate(itinerary, "moderate", {})
        assert any("driving > 4h" in e for e in result["errors"])

    def test_score_decreases_with_errors(self):
        itinerary = {
            "days": [{"day": 1,
                      "activities": [{"name": f"Place{i}"} for i in range(7)],
                      "total_distance_km": 200, "total_driving_hours": 1}]
        }
        result = _auto_validate(itinerary, "moderate", {})
        assert result["score"] < 1.0

    def test_intensive_pace_higher_limits(self):
        itinerary = {
            "days": [{"day": 1,
                      "activities": [{"name": f"Place{i}"} for i in range(5)],
                      "total_distance_km": 180, "total_driving_hours": 2}]
        }
        result = _auto_validate(itinerary, "intensive", {})
        assert result["is_valid"] is True

    def test_recommended_action_retry_on_errors(self):
        itinerary = {
            "days": [{"day": 1,
                      "activities": [{"name": "A"}, {"name": "B"}],
                      "total_distance_km": 300, "total_driving_hours": 1}]
        }
        result = _auto_validate(itinerary, "moderate", {})
        assert result["recommended_action"] == "retry"

    def test_recommended_action_proceed_on_no_errors(self, valid_itinerary):
        result = _auto_validate(valid_itinerary, "moderate", {})
        assert result["recommended_action"] == "proceed"


class TestNormalizePlaceName:
    def test_lowercase(self):
        assert _normalize_place_name("Batumi Boulevard") == "batumiboulevard"

    def test_removes_spaces_and_special_chars(self):
        result = _normalize_place_name("Café & Bar!")
        assert result == "cafbar"

    def test_cyrillic_kept(self):
        assert _normalize_place_name("Тбилиси") == "тбилиси"

    def test_duplicate_detection_works(self):
        """Normalization is enough to detect the same place with different casing."""
        a = _normalize_place_name("Batumi Boulevard")
        b = _normalize_place_name("BATUMI BOULEVARD")
        assert a == b
