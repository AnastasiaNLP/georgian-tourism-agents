# tests/conftest.py
"""Shared fixtures for all tests."""
# tests/conftest.py
"""Shared fixtures for all tests."""

from dotenv import load_dotenv

load_dotenv()

import pytest


@pytest.fixture(autouse=True)
def clear_runtime_settings_cache():
    """Settings are cached; tests that patch env need a fresh snapshot."""
    try:
        from config.settings import get_settings
        get_settings.cache_clear()
    except Exception:
        pass
    yield
    try:
        from config.settings import get_settings
        get_settings.cache_clear()
    except Exception:
        pass


# Test data

@pytest.fixture
def places_with_duplicates():
    """Places with RU/EN duplicates for deduplication tests."""
    return [
        {
            "name": "Batumi Botanical Garden",
            "location": "Batumi, Adjara",
            "category": "botanical garden",
            "score": 0.92,
            "description": "A beautiful botanical garden with exotic plants" * 3,
            "tags": ["nature", "garden"],
        },
        {
            "name": "Ботанический сад Батуми",
            "location": "Batumi, Adjara",
            "category": "botanical garden",
            "score": 0.89,
            "description": "Красивый ботанический сад",
            "tags": ["природа", "сад"],
        },
        {
            "name": "Mtirala National Park",
            "location": "Adjara",
            "category": "national park",
            "score": 0.85,
            "description": "Rainforest national park near Batumi",
            "tags": ["nature", "hiking"],
        },
        {
            "name": "Gonio Fortress",
            "location": "Gonio, Adjara",
            "category": "fortress",
            "score": 0.80,
            "description": "Ancient Roman fortress",
            "tags": ["history", "fortress"],
        },
    ]


@pytest.fixture
def places_geocoded():
    """Places with coordinates after GeoAgent processing."""
    return [
        {
            "name": "Batumi Boulevard",
            "location": "Batumi, Adjara",
            "category": "attraction",
            "lat": 41.6168,
            "lon": 41.6367,
            "description": "Main promenade",
            "tags": ["walking", "seafront"],
        },
        {
            "name": "Mtirala National Park",
            "location": "Adjara",
            "category": "national park",
            "lat": 41.7200,
            "lon": 41.8100,
            "description": "Rainforest",
            "tags": ["nature", "hiking"],
        },
        {
            "name": "Gonio Fortress",
            "location": "Gonio, Adjara",
            "category": "fortress",
            "lat": 41.5100,
            "lon": 41.6200,
            "description": "Ancient fortress",
            "tags": ["history"],
        },
    ]


@pytest.fixture
def distance_matrix():
    return {
        "Batumi Boulevard→Mtirala National Park": {"km": 25, "hours": 0.5},
        "Mtirala National Park→Batumi Boulevard": {"km": 25, "hours": 0.5},
        "Batumi Boulevard→Gonio Fortress": {"km": 15, "hours": 0.3},
        "Gonio Fortress→Batumi Boulevard": {"km": 15, "hours": 0.3},
        "Mtirala National Park→Gonio Fortress": {"km": 35, "hours": 0.7},
    }


@pytest.fixture
def valid_itinerary():
    return {
        "days": [
            {
                "day": 1,
                "location": "Batumi",
                "activities": [
                    {"name": "Batumi Boulevard", "location": "Batumi"},
                    {"name": "Batumi Tower", "location": "Batumi"},
                ],
                "total_distance_km": 5,
                "total_driving_hours": 0.1,
            },
            {
                "day": 2,
                "location": "Adjara",
                "activities": [
                    {"name": "Mtirala National Park", "location": "Adjara"},
                    {"name": "Gonio Fortress", "location": "Gonio"},
                ],
                "total_distance_km": 40,
                "total_driving_hours": 0.8,
            },
        ],
        "total_days": 2,
        "pace": "moderate",
        "summary": "2-day Adjara trip",
    }
