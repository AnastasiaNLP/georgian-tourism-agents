# tests/unit/test_search.py
"""Unit tests for agents/search/agent.py deduplication logic."""

import pytest
from agents.search.agent import _deduplicate_results


class TestDeduplication:
    def test_ru_en_duplicates_removed(self, places_with_duplicates):
        result = _deduplicate_results(places_with_duplicates)
        names = [p["name"] for p in result]
        botanical = [n for n in names if "botanical" in n.lower() or "ботанический" in n.lower()]
        assert len(botanical) == 1

    def test_keeps_longer_description(self, places_with_duplicates):
        result = _deduplicate_results(places_with_duplicates)
        kept = next(
            p for p in result
            if "botanical" in p["name"].lower() or "ботанический" in p["name"].lower()
        )
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
            {"name": "Place C", "location": "Gonio", "category": "fortress",
             "score": 0.80, "description": "x"},
        ]
        result = _deduplicate_results(places)
        assert len(result) == 2

    def test_score_too_far_not_deduped(self):
        places = [
            {"name": "Park A", "category": "national park", "score": 0.95,
             "description": "x", "location": "Adjara"},
            {"name": "Park B", "category": "national park", "score": 0.80,
             "description": "x", "location": "Kakheti"},
        ]
        result = _deduplicate_results(places)
        assert len(result) == 2


def test_no_hardcoded_translation_or_intent_filter():
    """Hardcoded Ru→En maps and keyword-based re-ranking must not exist in search agent."""
    import agents.search.agent as m
    assert not hasattr(m, "_translate_query"), \
        "_translate_query (hardcoded Ru→En translation) must be removed"
    assert not hasattr(m, "_filter_by_query_intent"), \
        "_filter_by_query_intent (hardcoded keyword re-ranking) must be removed"
    assert not hasattr(m, "_RU_CATEGORY_TO_EN"), \
        "_RU_CATEGORY_TO_EN (hardcoded category dict) must be removed"


def test_no_dead_web_tools():
    """search_web and get_place_details were never wired — must not exist in search_tools."""
    import src.tools.search_tools as m
    assert not hasattr(m, "search_web"), \
        "search_web was never called from any agent and must be removed"
    assert not hasattr(m, "get_place_details"), \
        "get_place_details was never called from any agent and must be removed"
