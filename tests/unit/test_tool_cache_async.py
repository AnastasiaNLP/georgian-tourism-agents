"""
Unit tests for async RedisCache and cached_* wrappers.

All network calls are mocked — no real Redis or external API is needed.
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# RedisCache — async get / set / delete
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_redis_cache_get_hit():
    """cache.get returns deserialized value when Redis responds with a result."""
    from src.tools.tool_cache import RedisCache

    cache = RedisCache.__new__(RedisCache)
    cache.url = "https://fake.upstash.io"
    cache.token = "token"
    cache._enabled = True
    cache._http_client = None

    fake_response = MagicMock()
    fake_response.json.return_value = {"result": json.dumps({"lat": 41.6, "lon": 41.6})}

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=fake_response)
    cache._http_client = mock_client

    result = await cache.get("geocode:abc123")

    assert result == {"lat": 41.6, "lon": 41.6}
    mock_client.get.assert_awaited_once()


@pytest.mark.asyncio
async def test_redis_cache_get_miss():
    """cache.get returns None when Redis has no entry for the key."""
    from src.tools.tool_cache import RedisCache

    cache = RedisCache.__new__(RedisCache)
    cache.url = "https://fake.upstash.io"
    cache.token = "token"
    cache._enabled = True
    cache._http_client = None

    fake_response = MagicMock()
    fake_response.json.return_value = {"result": None}

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=fake_response)
    cache._http_client = mock_client

    result = await cache.get("geocode:missing")

    assert result is None


@pytest.mark.asyncio
async def test_redis_cache_set_atomic_ttl():
    """
    cache.set sends exactly one POST request with EX query param.
    The old two-request pattern (SET + EXPIRE) must not appear.
    """
    from src.tools.tool_cache import RedisCache

    cache = RedisCache.__new__(RedisCache)
    cache.url = "https://fake.upstash.io"
    cache.token = "token"
    cache._enabled = True
    cache._http_client = None

    fake_response = MagicMock()
    fake_response.status_code = 200

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=fake_response)
    cache._http_client = mock_client

    await cache.set("search:xyz", [{"name": "Batumi"}], ttl_seconds=3600)

    # Exactly one POST — no separate EXPIRE call.
    assert mock_client.post.await_count == 1

    call_kwargs = mock_client.post.call_args
    url = call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs.get("url", "")
    params = call_kwargs.kwargs.get("params", {})
    assert "set/" in url, f"URL should contain 'set/': {url}"
    assert params.get("EX") == 3600, f"EX param must be 3600, got: {params}"


@pytest.mark.asyncio
async def test_redis_cache_disabled_get_returns_none():
    """When Redis is disabled, get() returns None without any HTTP call."""
    from src.tools.tool_cache import RedisCache

    cache = RedisCache.__new__(RedisCache)
    cache.url = ""
    cache.token = ""
    cache._enabled = False
    cache._http_client = None

    result = await cache.get("any_key")
    assert result is None
    assert cache._http_client is None  # client never created


@pytest.mark.asyncio
async def test_cached_search_qdrant_cache_hit():
    """
    On a cache hit, cached_search_qdrant returns cached results
    without calling the Qdrant search tool.
    """
    fake_results = [{"id": "1", "name": "Batumi"}]

    with patch("src.tools.tool_cache.get_cache") as mock_get_cache:
        mock_cache = AsyncMock()
        mock_cache.enabled = True
        mock_cache.get = AsyncMock(return_value=fake_results)
        mock_cache.set = AsyncMock()
        mock_get_cache.return_value = mock_cache

        with patch("src.tools.search_tools.search_qdrant") as mock_qdrant:
            from src.tools.tool_cache import cached_search_qdrant
            result = await cached_search_qdrant("churches tbilisi", top_k=5)

    assert result == fake_results
    mock_qdrant.invoke.assert_not_called()
    mock_cache.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_cached_search_qdrant_cache_miss_stores_result():
    """
    On a cache miss, cached_search_qdrant calls the tool and stores the result.
    """
    fake_results = [{"id": "2", "name": "Narikala"}]

    with patch("src.tools.tool_cache.get_cache") as mock_get_cache:
        mock_cache = AsyncMock()
        mock_cache.enabled = True
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()
        mock_get_cache.return_value = mock_cache

        with patch("src.tools.tool_cache.asyncio.to_thread", new_callable=AsyncMock,
                   return_value=fake_results):
            from src.tools.tool_cache import cached_search_qdrant
            result = await cached_search_qdrant("churches tbilisi", top_k=5)

    assert result == fake_results
    mock_cache.set.assert_awaited_once()


@pytest.mark.asyncio
async def test_cached_geocode_city_cache_miss_returns_tool_result():
    """
    On a cache miss (get returns None), cached_geocode_city calls the tool
    and returns its result. set() is called — disabled check is inside set() itself.
    """
    fake_coord = {"lat": 41.6168, "lon": 41.6367}

    with patch("src.tools.tool_cache.get_cache") as mock_get_cache:
        mock_cache = AsyncMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()
        mock_get_cache.return_value = mock_cache

        with patch("src.tools.tool_cache.asyncio.to_thread", new_callable=AsyncMock,
                   return_value=fake_coord):
            from src.tools.tool_cache import cached_geocode_city
            result = await cached_geocode_city("Batumi")

    assert result == fake_coord
    mock_cache.set.assert_awaited_once()
