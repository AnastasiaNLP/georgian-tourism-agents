"""Async Postgres connection pool for memory_manager."""
from __future__ import annotations

import logging
from typing import Optional

from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)

_pool: Optional[AsyncConnectionPool] = None


async def init_pool(conninfo: str) -> AsyncConnectionPool:
    """Explicitly initialize pool with given conninfo. Called once from lifespan."""
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            conninfo=conninfo,
            min_size=1,
            max_size=5,
            open=False,
        )
        await _pool.open()
        logger.info("memory: AsyncConnectionPool opened (max_size=5)")
    return _pool


async def get_pool() -> AsyncConnectionPool:
    """Return existing pool; lazy-init as fallback (tests / standalone scripts)."""
    if _pool is None:
        from config.settings import get_settings
        return await init_pool(get_settings().database_url)
    return _pool


async def close_pool() -> None:
    """Close pool on shutdown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("memory: AsyncConnectionPool closed")
