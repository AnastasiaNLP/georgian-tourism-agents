"""Unit tests for thread-level concurrency guard in api/router.py."""
import asyncio
import pytest

from api.router import _try_acquire_thread, _release_thread, _active_threads


@pytest.fixture(autouse=True)
async def clean_threads():
    _active_threads.clear()
    yield
    _active_threads.clear()


@pytest.mark.asyncio
async def test_acquire_free_thread():
    assert await _try_acquire_thread("t1") is True


@pytest.mark.asyncio
async def test_acquire_busy_thread_returns_false():
    await _try_acquire_thread("t2")
    assert await _try_acquire_thread("t2") is False


@pytest.mark.asyncio
async def test_release_allows_reacquire():
    await _try_acquire_thread("t3")
    await _release_thread("t3")
    assert await _try_acquire_thread("t3") is True


@pytest.mark.asyncio
async def test_release_nonexistent_does_not_raise():
    await _release_thread("nonexistent")  # must not raise


@pytest.mark.asyncio
async def test_independent_threads_do_not_block_each_other():
    assert await _try_acquire_thread("a") is True
    assert await _try_acquire_thread("b") is True
