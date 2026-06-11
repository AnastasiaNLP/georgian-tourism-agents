"""
Combined tests for Redis-backed rate limiter and thread lock.

Covers:
  - rate_limit_middleware: Redis path (allow / block), in-memory fallback, health probe exemption
  - _try_acquire_thread / _release_thread: Redis CAS path, in-memory fallback
  - RedisCache.acquire_lock: NX semantics, returns unique token
  - RedisCache.release_lock: compare-and-delete via Lua EVAL
  - _THREAD_LOCK_TTL: must exceed the pipeline agent-timeout budget
"""
from __future__ import annotations

import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import Request
from fastapi.responses import JSONResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(path: str = "/api/v1/plan", ip: str = "1.2.3.4") -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "query_string": b"",
        "headers": [],
        "client": (ip, 12345),
    }
    return Request(scope)


def _make_cache(enabled: bool, timestamps=None, acquire_token: str | None = "test-token"):
    cache = MagicMock()
    cache.enabled = enabled
    cache.get = AsyncMock(return_value=timestamps)
    cache.set = AsyncMock(return_value=True)
    cache.acquire_lock = AsyncMock(return_value=acquire_token)
    cache.release_lock = AsyncMock(return_value=True)
    return cache


async def _call_next(_):
    return MagicMock(status_code=200)


# ---------------------------------------------------------------------------
# Rate limiter — Redis path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rate_limit_redis_allows_under_limit():
    """Requests below MAX_RPM pass through when Redis is enabled."""
    from middleware.rate_limit import rate_limit_middleware, MAX_RPM
    cache = _make_cache(enabled=True, timestamps=[time.time() - 1] * (MAX_RPM - 1))

    with patch("middleware.rate_limit.get_cache", return_value=cache):
        resp = await rate_limit_middleware(_make_request(), _call_next)

    assert resp.status_code == 200
    cache.set.assert_awaited_once()


@pytest.mark.asyncio
async def test_rate_limit_redis_blocks_over_limit():
    """Returns 429 when Redis timestamps list is at MAX_RPM."""
    from middleware.rate_limit import rate_limit_middleware, MAX_RPM
    now = time.time()
    cache = _make_cache(enabled=True, timestamps=[now - i for i in range(MAX_RPM)])

    with patch("middleware.rate_limit.get_cache", return_value=cache):
        resp = await rate_limit_middleware(_make_request(), _call_next)

    assert resp.status_code == 429
    cache.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_rate_limit_redis_empty_key_allows():
    """First request from a new IP (key not in Redis) is allowed."""
    from middleware.rate_limit import rate_limit_middleware
    cache = _make_cache(enabled=True, timestamps=None)

    with patch("middleware.rate_limit.get_cache", return_value=cache):
        resp = await rate_limit_middleware(_make_request(), _call_next)

    assert resp.status_code == 200
    cache.set.assert_awaited_once()


@pytest.mark.asyncio
async def test_rate_limit_redis_old_timestamps_not_counted():
    """Timestamps older than 60s are discarded before counting."""
    from middleware.rate_limit import rate_limit_middleware, MAX_RPM
    old = [time.time() - 120] * MAX_RPM  # all expired
    cache = _make_cache(enabled=True, timestamps=old)

    with patch("middleware.rate_limit.get_cache", return_value=cache):
        resp = await rate_limit_middleware(_make_request(), _call_next)

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Rate limiter — in-memory fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rate_limit_fallback_allows_under_limit():
    """In-memory fallback passes requests under MAX_RPM."""
    import middleware.rate_limit as m
    from middleware.rate_limit import rate_limit_middleware
    m._requests.clear()
    m._warned_no_redis = False
    cache = _make_cache(enabled=False)

    with patch("middleware.rate_limit.get_cache", return_value=cache):
        resp = await rate_limit_middleware(_make_request(ip="9.9.9.9"), _call_next)

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_rate_limit_fallback_blocks_over_limit():
    """In-memory fallback returns 429 when limit is reached."""
    import middleware.rate_limit as m
    from middleware.rate_limit import rate_limit_middleware, MAX_RPM
    m._requests.clear()
    m._warned_no_redis = True  # suppress log
    cache = _make_cache(enabled=False)

    now = time.time()
    m._requests["8.8.8.8"] = [now - i for i in range(MAX_RPM)]

    with patch("middleware.rate_limit.get_cache", return_value=cache):
        resp = await rate_limit_middleware(_make_request(ip="8.8.8.8"), _call_next)

    assert resp.status_code == 429


@pytest.mark.asyncio
async def test_rate_limit_fallback_prunes_stale_ips():
    """IPs whose window is fully expired (stale or empty list) are evicted every 5 minutes."""
    import middleware.rate_limit as m
    from middleware.rate_limit import rate_limit_middleware

    m._requests.clear()
    m._warned_no_redis = True
    # Force the prune timer to trigger immediately on the next request.
    m._requests_last_prune = time.time() - 301

    stale_ip = "7.7.7.7"
    empty_ip = "6.6.6.6"
    # Stale entry: last timestamp is 120 seconds ago (well outside the 60s window).
    m._requests[stale_ip] = [time.time() - 120]
    # Empty entry: all timestamps were pruned, but the key still exists in the dict.
    m._requests[empty_ip] = []

    cache = _make_cache(enabled=False)
    with patch("middleware.rate_limit.get_cache", return_value=cache):
        # Any fresh request triggers the sweep.
        await rate_limit_middleware(_make_request(ip="5.5.5.5"), _call_next)

    assert stale_ip not in m._requests, "Stale IP must be evicted from _requests"
    assert empty_ip not in m._requests, "Empty-list IP must be evicted from _requests"


# ---------------------------------------------------------------------------
# Health probe exemption
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("path", [
    "/api/v1/health/live",
    "/api/v1/health/ready",
    "/api/v1/health",
    "/metrics",
    "/",
])
async def test_health_probes_exempt(path):
    """Exempt paths bypass rate limiting entirely — cache is never touched."""
    from middleware.rate_limit import rate_limit_middleware
    cache = _make_cache(enabled=True)

    with patch("middleware.rate_limit.get_cache", return_value=cache):
        resp = await rate_limit_middleware(_make_request(path=path), _call_next)

    assert resp.status_code == 200
    cache.get.assert_not_awaited()
    cache.set.assert_not_awaited()


# ---------------------------------------------------------------------------
# Thread lock — Redis path (token-based CAS)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_thread_lock_redis_acquire_success():
    """acquire_lock returning a token → _try_acquire_thread returns that token."""
    cache = _make_cache(enabled=True, acquire_token="abc123")

    with patch("api.router.get_cache", return_value=cache):
        from api.router import _try_acquire_thread
        result = await _try_acquire_thread("thread-abc")

    assert result == "abc123"
    cache.acquire_lock.assert_awaited_once_with("threadlock:thread-abc", ttl_seconds=360)


@pytest.mark.asyncio
async def test_thread_lock_redis_acquire_busy():
    """acquire_lock returning None → thread is already held."""
    cache = _make_cache(enabled=True, acquire_token=None)

    with patch("api.router.get_cache", return_value=cache):
        from api.router import _try_acquire_thread
        result = await _try_acquire_thread("thread-xyz")

    assert result is None


@pytest.mark.asyncio
async def test_thread_lock_redis_release_uses_cas():
    """_release_thread calls release_lock (compare-and-delete), not delete."""
    cache = _make_cache(enabled=True)

    with patch("api.router.get_cache", return_value=cache):
        from api.router import _release_thread
        await _release_thread("thread-abc", "mytoken")

    cache.release_lock.assert_awaited_once_with("threadlock:thread-abc", "mytoken")


# ---------------------------------------------------------------------------
# Thread lock — in-memory fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_thread_lock_fallback_acquire_and_release():
    """When Redis is disabled, in-memory lock still works correctly."""
    from api.router import _active_threads
    _active_threads.clear()
    cache = _make_cache(enabled=False)

    with patch("api.router.get_cache", return_value=cache):
        from api.router import _try_acquire_thread, _release_thread
        token = await _try_acquire_thread("t-fallback")
        assert token is not None
        assert await _try_acquire_thread("t-fallback") is None
        await _release_thread("t-fallback", token)
        assert await _try_acquire_thread("t-fallback") is not None

    _active_threads.clear()


# ---------------------------------------------------------------------------
# _THREAD_LOCK_TTL — must outlive the pipeline budget
# ---------------------------------------------------------------------------

def test_thread_lock_ttl_exceeds_pipeline_budget():
    """Lock TTL must exceed the sum of all core agent timeouts to survive slow runs."""
    from api.router import _THREAD_LOCK_TTL
    from agents.error_handler import AGENT_TIMEOUTS

    core_agents = ["search_agent", "geo_agent", "planning_agent", "validation_agent", "response_agent"]
    pipeline_budget = sum(AGENT_TIMEOUTS[a] for a in core_agents)

    assert _THREAD_LOCK_TTL > pipeline_budget, (
        f"Lock TTL ({_THREAD_LOCK_TTL}s) must exceed pipeline budget "
        f"({pipeline_budget}s = "
        + " + ".join(f"{a.split('_')[0]}({AGENT_TIMEOUTS[a]})" for a in core_agents)
        + ")"
    )


# ---------------------------------------------------------------------------
# RedisCache.acquire_lock — NX semantics, returns token string
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_acquire_lock_returns_token_on_ok():
    """acquire_lock returns a non-empty token string when Upstash responds OK."""
    from src.tools.tool_cache import RedisCache

    cache = RedisCache.__new__(RedisCache)
    cache.url = "https://fake.upstash.io"
    cache.token = "token"
    cache._enabled = True
    cache._http_client = None

    fake_response = MagicMock()
    fake_response.json.return_value = {"result": "OK"}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=fake_response)
    cache._http_client = mock_client

    result = await cache.acquire_lock("threadlock:t1", ttl_seconds=360)

    assert result is not None and len(result) > 0
    call_kwargs = mock_client.post.call_args.kwargs
    assert call_kwargs["params"].get("NX") == ""
    assert call_kwargs["params"].get("EX") == 360
    # The returned token is what was written to Redis
    assert call_kwargs["content"] == result


@pytest.mark.asyncio
async def test_acquire_lock_returns_none_when_key_exists():
    """acquire_lock returns None when Upstash responds with result=null (lock held)."""
    from src.tools.tool_cache import RedisCache

    cache = RedisCache.__new__(RedisCache)
    cache.url = "https://fake.upstash.io"
    cache.token = "token"
    cache._enabled = True
    cache._http_client = None

    fake_response = MagicMock()
    fake_response.json.return_value = {"result": None}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=fake_response)
    cache._http_client = mock_client

    result = await cache.acquire_lock("threadlock:t1", ttl_seconds=360)

    assert result is None


@pytest.mark.asyncio
async def test_acquire_lock_disabled_returns_none():
    """acquire_lock returns None without any HTTP call when Redis is disabled."""
    from src.tools.tool_cache import RedisCache

    cache = RedisCache.__new__(RedisCache)
    cache._enabled = False
    cache._http_client = None

    result = await cache.acquire_lock("any_key")

    assert result is None
    assert cache._http_client is None


# ---------------------------------------------------------------------------
# RedisCache.release_lock — compare-and-delete via Lua EVAL
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_release_lock_calls_eval_with_token():
    """release_lock invokes EVAL with the correct key and token."""
    from src.tools.tool_cache import RedisCache

    cache = RedisCache.__new__(RedisCache)
    cache.url = "https://fake.upstash.io"
    cache.token = "bearer"
    cache._enabled = True
    cache._http_client = None

    fake_response = MagicMock()
    fake_response.json.return_value = {"result": 1}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=fake_response)
    cache._http_client = mock_client

    result = await cache.release_lock("threadlock:t1", "mytoken123")

    assert result is True
    call_url = mock_client.post.call_args.args[0]
    assert "eval" in call_url
    # Token must appear in the URL path (URL-encoded)
    assert "mytoken123" in call_url


@pytest.mark.asyncio
async def test_release_lock_returns_false_on_token_mismatch():
    """release_lock returns False when EVAL returns 0 (token does not match)."""
    from src.tools.tool_cache import RedisCache

    cache = RedisCache.__new__(RedisCache)
    cache.url = "https://fake.upstash.io"
    cache.token = "bearer"
    cache._enabled = True

    fake_response = MagicMock()
    fake_response.json.return_value = {"result": 0}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=fake_response)
    cache._http_client = mock_client

    result = await cache.release_lock("threadlock:t1", "wrong-token")

    assert result is False


@pytest.mark.asyncio
async def test_release_lock_disabled_returns_false():
    """release_lock returns False without any HTTP call when Redis is disabled."""
    from src.tools.tool_cache import RedisCache

    cache = RedisCache.__new__(RedisCache)
    cache._enabled = False
    cache._http_client = None

    result = await cache.release_lock("any_key", "any_token")

    assert result is False
    assert cache._http_client is None
