# tests/unit/test_validation.py
"""Unit tests for agents/validation/agent.py and _auto_validate."""

import pytest
from agents.validation.agent import (
    _auto_validate,
    _haversine_km,
    _normalize_place_name,
)


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


    def test_auto_validate_uses_settings_thresholds(self):
        """_auto_validate reads km/acts/driving limits from settings, not hardcoded values."""
        from unittest.mock import patch, MagicMock

        mock_settings = MagicMock()
        mock_settings.validation_max_km_moderate = 50   # lower than default 150
        mock_settings.validation_max_km_relaxed = 40
        mock_settings.validation_max_km_intensive = 60
        mock_settings.validation_max_acts_moderate = 3
        mock_settings.validation_max_acts_relaxed = 2
        mock_settings.validation_max_acts_intensive = 5
        mock_settings.validation_max_driving_hours = 2

        itinerary = {
            "days": [{"day": 1,
                      "activities": [{"name": "A"}, {"name": "B"}],
                      "total_distance_km": 80,  # > custom limit 50, < default 150
                      "total_driving_hours": 1}]
        }
        with patch("config.settings.get_settings", return_value=mock_settings):
            result = _auto_validate(itinerary, "moderate", {})

        assert result["is_valid"] is False
        assert any("exceeds" in e for e in result["errors"])


class TestDistanceGrounding:
    """Reported distances are cross-checked against measured route data."""

    def test_under_report_via_matrix_is_error(self):
        """Reported km far below the measured driving distance → error → retry."""
        itinerary = {
            "days": [{"day": 1,
                      "activities": [{"name": "Place A"}, {"name": "Place B"}],
                      "total_distance_km": 10, "total_driving_hours": 1}]
        }
        distances = {"Place A→Place B": {"km": 200, "hours": 3}}
        result = _auto_validate(itinerary, "moderate", distances)
        assert result["is_valid"] is False
        assert any("measured route" in e for e in result["errors"])
        assert result["recommended_action"] == "retry"

    def test_under_report_via_haversine_is_error(self):
        """With no matrix entry, the straight-line lower bound still catches under-reporting."""
        itinerary = {
            "days": [{"day": 1,
                      "activities": [{"name": "Place A"}, {"name": "Place B"}],
                      "total_distance_km": 10, "total_driving_hours": 1}]
        }
        # Tbilisi ~ Batumi: ~270 km straight line.
        coords = {
            "placea": (41.72, 44.79),
            "placeb": (41.64, 41.64),
        }
        result = _auto_validate(itinerary, "moderate", {}, coords)
        assert result["is_valid"] is False
        assert any("measured route" in e for e in result["errors"])

    def test_accurate_plan_passes(self):
        """Reported distance at or above the measured route is accepted."""
        itinerary = {
            "days": [{"day": 1,
                      "activities": [{"name": "Place A"}, {"name": "Place B"}],
                      "total_distance_km": 110, "total_driving_hours": 1.5}]
        }
        distances = {"Place A→Place B": {"km": 100, "hours": 1.4}}
        coords = {"placea": (41.72, 44.79), "placeb": (41.79, 44.82)}
        result = _auto_validate(itinerary, "moderate", distances, coords)
        assert result["is_valid"] is True
        assert result["low_distance_confidence"] is False

    def test_low_grounding_flags_degrade(self):
        """Geo data exists but covers none of the day's legs → low confidence flag."""
        itinerary = {
            "days": [{"day": 1,
                      "activities": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
                      "total_distance_km": 30, "total_driving_hours": 1}]
        }
        # Matrix is non-empty (geo ran) but has no entry for this day's pairs,
        # and no coordinates are supplied for A/B/C.
        distances = {"Foo→Bar": {"km": 10, "hours": 0.2}}
        result = _auto_validate(itinerary, "moderate", distances)
        assert result["low_distance_confidence"] is True
        assert result["distance_grounding"] == 0.0

    def test_no_geo_data_no_false_degrade(self):
        """No matrix and no coordinates (e.g. geo step skipped) → no low-confidence flag."""
        itinerary = {
            "days": [{"day": 1,
                      "activities": [{"name": "A"}, {"name": "B"}],
                      "total_distance_km": 30, "total_driving_hours": 1}]
        }
        result = _auto_validate(itinerary, "moderate", {})
        assert result["low_distance_confidence"] is False


class TestHaversine:
    def test_known_distance(self):
        """Tbilisi → Batumi is roughly 270 km in a straight line."""
        km = _haversine_km(41.72, 44.79, 41.64, 41.64)
        assert 250 < km < 290

    def test_zero_for_same_point(self):
        assert _haversine_km(41.7, 44.8, 41.7, 44.8) == pytest.approx(0.0, abs=1e-6)


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
