# tests/unit/test_search.py
"""Unit tests for agents/search/agent.py deduplication and intent filtering."""

import pytest
from agents.search.agent import _deduplicate_results, _filter_by_query_intent, _translate_query


class TestDeduplication:
    def test_ru_en_duplicates_removed(self, places_with_duplicates):
        result = _deduplicate_results(places_with_duplicates)
        names = [p["name"] for p in result]
        # One of the two botanical garden entries should remain.
        botanical = [n for n in names if "botanical" in n.lower() or "ботанический" in n.lower()]
        assert len(botanical) == 1

    def test_keeps_longer_description(self, places_with_duplicates):
        result = _deduplicate_results(places_with_duplicates)
        kept = next(
            p for p in result
            if "botanical" in p["name"].lower() or "ботанический" in p["name"].lower()
        )
        # The EN version is longer and should be kept.
        assert "Batumi Botanical Garden" == kept["name"]

    def test_non_duplicates_kept(self, places_with_duplicates):
        result = _deduplicate_results(places_with_duplicates)
        names = [p["name"] for p in result]
        assert "Mtirala National Park" in names
        assert "Gonio Fortress" in names

    def test_empty_list(self):
        assert _deduplicate_results([]) == []

    def test_single_item(self):
        places = [{"name": "A", "category": "park", "score": 0.9, "description": "x"}]
        assert _deduplicate_results(places) == places

    def test_same_location_deduped(self):
        places = [
            {"name": "Place A", "location": "Batumi", "category": "museum",
             "score": 0.9, "description": "long " * 10},
            {"name": "Place B", "location": "Batumi", "category": "park",
             "score": 0.85, "description": "short"},
            # Different location, not a duplicate.
            {"name": "Place C", "location": "Gonio", "category": "fortress",
             "score": 0.80, "description": "x"},
        ]
        result = _deduplicate_results(places)
        # Place A and Place B have the same location, so one is removed.
        assert len(result) == 2

    def test_score_too_far_not_deduped(self):
        """If score differs by more than 0.08 and location differs, keep both."""
        places = [
            {"name": "Park A", "category": "national park", "score": 0.95,
             "description": "x", "location": "Adjara"},
            {"name": "Park B", "category": "national park", "score": 0.80,
             "description": "x", "location": "Kakheti"},
        ]
        result = _deduplicate_results(places)
        assert len(result) == 2


class TestIntentFilter:
    def test_nature_query_boosts_nature(self):
        places = [
            {"name": "Museum", "category": "museum", "score": 0.9,
             "tags": ["museum"], "description": "", "location": ""},
            {"name": "National Park", "category": "national park", "score": 0.85,
             "tags": ["nature", "hiking"], "description": "", "location": ""},
        ]
        result = _filter_by_query_intent(places, "природа и горы")
        assert result[0]["name"] == "National Park"

    def test_culture_query_boosts_museums(self):
        places = [
            {"name": "Beach", "category": "beach", "score": 0.9,
             "tags": ["beach"], "description": "", "location": ""},
            {"name": "History Museum", "category": "museum", "score": 0.85,
             "tags": ["museum", "history"], "description": "", "location": ""},
        ]
        result = _filter_by_query_intent(places, "история и культура")
        assert result[0]["name"] == "History Museum"

    def test_no_keywords_unchanged(self, places_with_duplicates):
        result = _filter_by_query_intent(places_with_duplicates, "Батуми 3 дня")
        assert result == places_with_duplicates

    def test_empty_query_unchanged(self, places_with_duplicates):
        result = _filter_by_query_intent(places_with_duplicates, "")
        assert result == places_with_duplicates


class TestTranslateQuery:
    def test_russian_nature_words(self):
        result = _translate_query("природа и горы")
        assert "nature" in result
        assert "mountains" in result

    def test_russian_culture_words(self):
        result = _translate_query("музей и крепость")
        assert "museum" in result
        assert "fortress" in result

    def test_english_unchanged(self):
        result = _translate_query("hiking in mountains")
        assert "hiking" in result
        assert "mountains" in result
