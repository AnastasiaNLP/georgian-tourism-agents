"""
Unit tests for async Qdrant operations in MemoryManager.
All Qdrant I/O is mocked — no real Qdrant connection needed.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_manager():
    """Return a fresh MemoryManager without calling get_memory_manager() singleton."""
    from src.memory.memory_manager import MemoryManager
    return MemoryManager()


@pytest.mark.asyncio
async def test_save_episode_awaits_upsert():
    """save_episode must await AsyncQdrantClient.upsert."""
    mm = _make_manager()

    existing = MagicMock()
    existing.name = "user_memory_episodes_v1"

    mock_client = AsyncMock()
    mock_client.get_collections.return_value = MagicMock(collections=[existing])
    mock_client.upsert = AsyncMock(return_value=None)
    mm._async_qdrant = mock_client

    with patch.object(mm, "_embed", return_value=[0.1] * 768):
        result = await mm.save_episode(
            user_id="u1",
            episode={
                "task": "trip to Batumi",
                "plan_summary": "2-day Adjara",
                "places": ["Batumi Boulevard"],
                "result": "success",
            },
        )

    assert result is True
    mock_client.upsert.assert_awaited_once()


@pytest.mark.asyncio
async def test_retrieve_episodes_returns_empty_on_error():
    """Exception during query_points → retrieve_episodes returns [] without raising."""
    mm = _make_manager()

    mock_client = AsyncMock()
    mock_client.get_collections.return_value = MagicMock(collections=[])
    mock_client.create_collection = AsyncMock()
    mock_client.create_payload_index = AsyncMock()
    mock_client.query_points = AsyncMock(side_effect=ConnectionError("Qdrant unreachable"))
    mm._async_qdrant = mock_client

    with patch.object(mm, "_embed", return_value=[0.1] * 768):
        episodes = await mm.retrieve_episodes(user_id="u1", query="churches tbilisi")

    assert episodes == []


@pytest.mark.asyncio
async def test_ensure_collection_creates_when_missing():
    """_ensure_episode_collection creates the collection when it does not exist."""
    from src.memory.memory_manager import EPISODE_COLLECTION
    mm = _make_manager()

    mock_client = AsyncMock()
    mock_client.get_collections.return_value = MagicMock(collections=[])
    mock_client.create_collection = AsyncMock()
    mock_client.create_payload_index = AsyncMock()
    mm._async_qdrant = mock_client

    await mm._ensure_episode_collection()

    mock_client.create_collection.assert_awaited_once()
    call_kwargs = mock_client.create_collection.call_args.kwargs
    assert call_kwargs.get("collection_name") == EPISODE_COLLECTION

    mock_client.create_payload_index.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_collection_skips_when_exists():
    """_ensure_episode_collection must NOT recreate an existing collection."""
    from src.memory.memory_manager import EPISODE_COLLECTION
    mm = _make_manager()

    existing = MagicMock()
    existing.name = EPISODE_COLLECTION  # MagicMock(name=...) sets mock name, not .name attr

    mock_client = AsyncMock()
    mock_client.get_collections.return_value = MagicMock(collections=[existing])
    mock_client.create_collection = AsyncMock()
    mm._async_qdrant = mock_client

    await mm._ensure_episode_collection()

    mock_client.create_collection.assert_not_awaited()


@pytest.mark.asyncio
async def test_async_qdrant_client_has_timeout():
    """_get_async_qdrant builds the client with timeout=10."""
    mm = _make_manager()

    fake_client = MagicMock()
    with patch("src.memory.memory_manager.AsyncQdrantClient", return_value=fake_client) as mock_cls:
        mm._get_async_qdrant()

    _, kwargs = mock_cls.call_args
    assert kwargs.get("timeout") == 10


@pytest.mark.asyncio
async def test_aclose_calls_client_close():
    """aclose() must close the underlying AsyncQdrantClient and clear the reference."""
    mm = _make_manager()

    mock_client = AsyncMock()
    mock_client.close = AsyncMock()
    mm._async_qdrant = mock_client

    await mm.aclose()

    mock_client.close.assert_awaited_once()
    assert mm._async_qdrant is None


@pytest.mark.asyncio
async def test_aclose_noop_when_no_client():
    """aclose() must not raise when the client was never initialized."""
    mm = _make_manager()
    assert mm._async_qdrant is None
    await mm.aclose()  # must not raise


# ---------------------------------------------------------------------------
# memory_load_node — episode retrieval wired into graph node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_memory_load_node_loads_profile():
    """memory_load_node must load profile and store it in user_memory."""
    from graph.main_graph import memory_load_node

    fake_profile = {"user_id": "u1", "visited_places": ["Batumi"], "past_trips": []}
    mock_mm = AsyncMock()
    mock_mm.load_profile = AsyncMock(return_value=dict(fake_profile))

    state = {
        "request_id": "mem-load-test",
        "user_id": "u1",
        "user_query": "plan my Georgia trip",
        "execution_mode": "normal",
        "agent_history": [],
        "errors": [],
    }

    with patch("src.memory.memory_manager.get_memory_manager", return_value=mock_mm):
        result = await memory_load_node(state)

    mock_mm.load_profile.assert_awaited_once_with("u1")
    assert result["user_memory"]["visited_places"] == ["Batumi"]


@pytest.mark.asyncio
async def test_memory_load_node_defers_episode_retrieval():
    """memory_load_node must NOT call retrieve_episodes — retrieval is deferred to planning_agent."""
    from graph.main_graph import memory_load_node

    mock_mm = AsyncMock()
    mock_mm.load_profile = AsyncMock(return_value={"user_id": "u1"})

    state = {
        "request_id": "defer-test",
        "user_id": "u1",
        "user_query": "Adjara 2 days",
        "execution_mode": "normal",
        "agent_history": [],
        "errors": [],
    }

    with patch("src.memory.memory_manager.get_memory_manager", return_value=mock_mm):
        await memory_load_node(state)

    mock_mm.retrieve_episodes.assert_not_awaited()


@pytest.mark.asyncio
async def test_memory_load_node_anonymous_skips_profile():
    """Anonymous requests (no user_id) must return user_memory=None without calling memory manager."""
    from graph.main_graph import memory_load_node

    mock_mm = AsyncMock()

    state = {
        "request_id": "anon-test",
        "user_id": None,
        "user_query": "plan my Georgia trip",
        "execution_mode": "normal",
        "agent_history": [],
        "errors": [],
    }

    with patch("src.memory.memory_manager.get_memory_manager", return_value=mock_mm):
        result = await memory_load_node(state)

    mock_mm.load_profile.assert_not_awaited()
    assert result["user_memory"] is None
