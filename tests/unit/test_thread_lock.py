"""Unit tests for thread-level concurrency guard in api/router.py — in-memory path."""
import asyncio
import pytest
from unittest.mock import patch, MagicMock

from api.router import _try_acquire_thread, _release_thread, _active_threads


def _disabled_cache():
    c = MagicMock()
    c.enabled = False
    return c


@pytest.fixture(autouse=True)
async def clean_threads():
    _active_threads.clear()
    yield
    _active_threads.clear()


@pytest.fixture(autouse=True)
def force_inmemory(clean_threads):
    """Force in-memory path so tests are not affected by Redis availability."""
    with patch("api.router.get_cache", return_value=_disabled_cache()):
        yield


@pytest.mark.asyncio
async def test_acquire_free_thread():
    token = await _try_acquire_thread("t1")
    assert token is not None


@pytest.mark.asyncio
async def test_acquire_busy_thread_returns_none():
    await _try_acquire_thread("t2")
    assert await _try_acquire_thread("t2") is None


@pytest.mark.asyncio
async def test_release_allows_reacquire():
    token = await _try_acquire_thread("t3")
    await _release_thread("t3", token)
    assert await _try_acquire_thread("t3") is not None


@pytest.mark.asyncio
async def test_release_nonexistent_does_not_raise():
    await _release_thread("nonexistent", "__inmem__")  # must not raise


@pytest.mark.asyncio
async def test_independent_threads_do_not_block_each_other():
    assert await _try_acquire_thread("a") is not None
    assert await _try_acquire_thread("b") is not None
