"""Rate limiter middleware.

When Redis is available: sliding-window counter is stored per IP in Redis,
shared across all Uvicorn workers.

When Redis is disabled: falls back to an in-memory dict (single-worker only).
A warning is emitted once to make the limitation visible in logs.
"""
import time
import logging
from collections import defaultdict
from fastapi import Request
from fastapi.responses import JSONResponse
from config.settings import get_settings
from src.tools.tool_cache import get_cache

logger = logging.getLogger(__name__)

_requests: dict = defaultdict(list)
MAX_RPM = get_settings().rate_limit_rpm

# Paths that must never consume rate-limit slots.
_EXEMPT_PATHS = frozenset({
    "/",
    "/metrics",
    "/docs",
    "/openapi.json",
    "/api/v1/health",
    "/api/v1/health/live",
    "/api/v1/health/ready",
})

_warned_no_redis = False


async def rate_limit_middleware(request: Request, call_next):
    """Allow at most MAX_RPM requests per minute from one IP."""
    global _warned_no_redis

    if request.url.path in _EXEMPT_PATHS:
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    cache = get_cache()

    if cache.enabled:
        key = f"ratelimit:{client_ip}"
        stored = await cache.get(key)
        timestamps = [t for t in (stored or []) if now - t < 60]
        if len(timestamps) >= MAX_RPM:
            logger.warning(f"Rate limit exceeded for {client_ip} (redis)")
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit exceeded. Max {MAX_RPM} requests per minute."},
            )
        timestamps.append(now)
        await cache.set(key, timestamps, ttl_seconds=60)
    else:
        if not _warned_no_redis:
            logger.warning(
                "Redis not configured — rate limiter uses in-memory state "
                "(not shared across workers)"
            )
            _warned_no_redis = True

        _requests[client_ip] = [t for t in _requests[client_ip] if now - t < 60]
        if len(_requests[client_ip]) >= MAX_RPM:
            logger.warning(f"Rate limit exceeded for {client_ip} (in-memory)")
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit exceeded. Max {MAX_RPM} requests per minute."},
            )
        _requests[client_ip].append(now)

    return await call_next(request)
