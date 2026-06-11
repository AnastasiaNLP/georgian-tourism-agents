"""Async Postgres connection pool for memory_manager."""
from __future__ import annotations

import logging
from typing import Optional

from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)

_pool: Optional[AsyncConnectionPool] = None


async def init_pool(conninfo: str) -> AsyncConnectionPool:
    """Explicitly initialize pool with given conninfo. Called once from lifespan.

    kwargs mirror AsyncPostgresSaver.from_conn_string requirements:
    - autocommit=True so checkpointer migrations and aput writes commit immediately
      without relying on commit-on-return pool semantics.
    - prepare_threshold=0 to match the saver's expected connection mode.
    row_factory is intentionally left as the psycopg default (tuple) so that
    memory_manager's row[0] access works; the saver sets dict_row per-cursor.
    """
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            conninfo=conninfo,
            min_size=1,
            max_size=5,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        await _pool.open()
        logger.info("memory: AsyncConnectionPool opened (max_size=5, autocommit=True)")
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
