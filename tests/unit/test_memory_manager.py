"""Unit tests for memory_manager pure functions and intent guard."""
import time
import pytest

from src.memory.memory_manager import (
    empty_profile,
    merge_profile,
    MAX_INTERESTS,
    MAX_VISITED,
    MAX_TRIPS,
    MemoryManager,
)


# ---------------------------------------------------------------------------
# empty_profile
# ---------------------------------------------------------------------------

def test_empty_profile_structure():
    p = empty_profile("user-1")
    assert p["user_id"] == "user-1"
    assert p["preferred_pace"] is None
    assert p["preferred_language"] is None
    assert p["interests"] == []
    assert p["visited_places"] == []
    assert p["past_trips"] == []
    assert "created_at" in p
    assert "updated_at" in p


# ---------------------------------------------------------------------------
# merge_profile
# ---------------------------------------------------------------------------

def test_merge_pace_overwrites():
    old = empty_profile("u1")
    old["preferred_pace"] = "relaxed"
    merged = merge_profile(old, {"preferred_pace": "intensive"})
    assert merged["preferred_pace"] == "intensive"


def test_merge_pace_absent_keeps_old():
    old = empty_profile("u1")
    old["preferred_pace"] = "relaxed"
    merged = merge_profile(old, {"preferred_language": "en"})
    assert merged["preferred_pace"] == "relaxed"


def test_merge_language_overwrites():
    old = empty_profile("u1")
    old["preferred_language"] = "ru"
    merged = merge_profile(old, {"preferred_language": "en"})
    assert merged["preferred_language"] == "en"


def test_merge_visited_places_dedup():
    old = empty_profile("u1")
    old["visited_places"] = ["Tbilisi", "Batumi"]
    merged = merge_profile(old, {"visited_places": ["Batumi", "Kutaisi"]})
    assert sorted(merged["visited_places"]) == ["Batumi", "Kutaisi", "Tbilisi"]


def test_merge_visited_places_cap():
    old = empty_profile("u1")
    old["visited_places"] = [f"place_{i}" for i in range(MAX_VISITED)]
    merged = merge_profile(old, {"visited_places": ["new_place"]})
    assert len(merged["visited_places"]) <= MAX_VISITED


def test_merge_interests_dedup():
    old = empty_profile("u1")
    old["interests"] = ["history", "nature"]
    merged = merge_profile(old, {"interests": ["nature", "food"]})
    assert sorted(merged["interests"]) == ["food", "history", "nature"]


def test_merge_interests_cap():
    old = empty_profile("u1")
    old["interests"] = [f"tag_{i}" for i in range(MAX_INTERESTS)]
    merged = merge_profile(old, {"interests": ["new_tag"]})
    assert len(merged["interests"]) <= MAX_INTERESTS


def test_merge_past_trips_rotation():
    old = empty_profile("u1")
    old["past_trips"] = [{"date": f"2024-0{i+1}-01"} for i in range(MAX_TRIPS)]
    new_trip = {"date": "2025-01-01", "query": "new trip"}
    merged = merge_profile(old, {"new_trip": new_trip})
    assert len(merged["past_trips"]) == MAX_TRIPS
    assert merged["past_trips"][-1] == new_trip


def test_merge_past_trips_append_when_under_cap():
    old = empty_profile("u1")
    old["past_trips"] = [{"date": "2024-01-01"}]
    new_trip = {"date": "2025-01-01"}
    merged = merge_profile(old, {"new_trip": new_trip})
    assert len(merged["past_trips"]) == 2
    assert merged["past_trips"][-1] == new_trip


def test_merge_empty_new_data_unchanged():
    old = empty_profile("u1")
    old["preferred_pace"] = "moderate"
    old["visited_places"] = ["Tbilisi"]
    merged = merge_profile(old, {})
    assert merged["preferred_pace"] == "moderate"
    assert merged["visited_places"] == ["Tbilisi"]


def test_merge_updated_at_changes():
    old = empty_profile("u1")
    original_ts = old["updated_at"]
    time.sleep(0.01)
    merged = merge_profile(old, {"preferred_pace": "relaxed"})
    assert merged["updated_at"] >= original_ts


# ---------------------------------------------------------------------------
# extract_profile_update — intent guard
# ---------------------------------------------------------------------------

def _make_state(intent: str, with_itinerary: bool = True) -> dict:
    itinerary = None
    if with_itinerary:
        itinerary = {
            "days": [
                {
                    "day": 1,
                    "activities": [
                        {"name": "Narikala", "location": "Tbilisi"},
                        {"name": "Old Town", "location": "Tbilisi"},
                    ],
                }
            ]
        }
    return {
        "intent": intent,
        "user_language": "ru",
        "trip_parameters": {"pace": "moderate"},
        "raw_itinerary": itinerary,
        "enriched_itinerary": None,
        "user_query": "test query",
        "execution_mode": "normal",
    }


def test_extract_visited_places_plan_intent():
    mm = MemoryManager()
    update = mm.extract_profile_update(_make_state("PLAN"))
    assert "visited_places" in update
    assert len(update["visited_places"]) > 0


def test_extract_visited_places_revise_intent():
    mm = MemoryManager()
    update = mm.extract_profile_update(_make_state("REVISE"))
    assert "visited_places" in update


def test_extract_visited_places_info_excluded():
    mm = MemoryManager()
    update = mm.extract_profile_update(_make_state("INFO"))
    assert "visited_places" not in update, "INFO must not accumulate visited_places"


def test_extract_visited_places_consult_excluded():
    mm = MemoryManager()
    update = mm.extract_profile_update(_make_state("CONSULT"))
    assert "visited_places" not in update


def test_extract_new_trip_info_excluded():
    mm = MemoryManager()
    update = mm.extract_profile_update(_make_state("INFO"))
    assert "new_trip" not in update, "INFO must not write new_trip"


def test_extract_new_trip_consult_excluded():
    mm = MemoryManager()
    update = mm.extract_profile_update(_make_state("CONSULT"))
    assert "new_trip" not in update


def test_extract_new_trip_present_for_plan():
    mm = MemoryManager()
    update = mm.extract_profile_update(_make_state("PLAN"))
    assert "new_trip" in update
    assert update["new_trip"]["intent"] == "PLAN"


def test_extract_language_always_written():
    mm = MemoryManager()
    update = mm.extract_profile_update(_make_state("INFO", with_itinerary=False))
    assert update.get("preferred_language") == "ru"


def test_extract_pace_always_written():
    mm = MemoryManager()
    update = mm.extract_profile_update(_make_state("INFO", with_itinerary=False))
    assert update.get("preferred_pace") == "moderate"


# ---------------------------------------------------------------------------
# cleanup_stale_profiles
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cleanup_stale_profiles_deletes_old_rows():
    """cleanup_stale_profiles runs a DELETE and returns the row count."""
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_cursor = MagicMock()
    mock_cursor.rowcount = 3

    mock_conn = AsyncMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)
    mock_conn.execute = AsyncMock(return_value=mock_cursor)

    mock_pool = AsyncMock()
    mock_pool.connection = MagicMock(return_value=mock_conn)

    mm = MemoryManager()
    with patch("src.memory.db.get_pool", AsyncMock(return_value=mock_pool)):
        deleted = await mm.cleanup_stale_profiles(days=90)

    assert deleted == 3
    mock_conn.execute.assert_awaited_once()
    sql, params = mock_conn.execute.call_args.args
    assert "DELETE" in sql.upper()
    assert params == (90,)


@pytest.mark.asyncio
async def test_cleanup_stale_profiles_returns_zero_on_error():
    """Database failure during cleanup must not raise — returns 0."""
    from unittest.mock import AsyncMock, patch

    mm = MemoryManager()
    with patch("src.memory.db.get_pool", AsyncMock(side_effect=ConnectionError("db down"))):
        deleted = await mm.cleanup_stale_profiles()

    assert deleted == 0
