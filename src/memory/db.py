"""Async Postgres connection pool for memory_manager."""
from __future__ import annotations

import logging
from typing import Optional

from psycopg_pool import AsyncConnectionPool

from config.settings import get_settings

logger = logging.getLogger(__name__)

_pool: Optional[AsyncConnectionPool] = None


async def get_pool() -> AsyncConnectionPool:
    """Lazy singleton pool for memory storage."""
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = AsyncConnectionPool(
            conninfo=settings.database_url,
            min_size=1,
            max_size=5,
            open=False,
        )
        await _pool.open()
        logger.info("memory: AsyncConnectionPool opened (max_size=5)")
    return _pool


async def close_pool() -> None:
    """Close pool on shutdown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("memory: AsyncConnectionPool closed")
