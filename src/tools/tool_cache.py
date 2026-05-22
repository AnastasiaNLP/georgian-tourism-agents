"""
Retry and cache helpers for external tools.

TTL strategy:
- search cache: 1 hour
- distance cache: 24 hours
- weather cache: 3 hours
"""

import json
import hashlib
import logging
import asyncio
import httpx
from typing import Any, Optional, Callable
from functools import wraps
from config.settings import get_settings

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Upstash Redis over HTTP API.
    """

    def __init__(self):
        settings = get_settings()
        self.url = settings.upstash_redis_url or ""
        self.token = settings.upstash_redis_token or ""
        self._enabled = bool(self.url and self.token)

        if not self._enabled:
            logger.warning("Redis cache disabled — UPSTASH_REDIS_URL or TOKEN not set")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache. Returns None when missing."""
        if not self._enabled:
            return None
        try:
            r = httpx.get(
                f"{self.url}/get/{key}",
                headers=self._headers(),
                timeout=2.0
            )
            data = r.json()
            result = data.get("result")
            if result is None:
                return None
            return json.loads(result)
        except Exception as e:
            logger.debug(f"Cache GET failed for {key}: {e}")
            return None

    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Save a value with TTL."""
        if not self._enabled:
            return False
        try:
            serialized = json.dumps(value, ensure_ascii=False)
            r = httpx.post(
                f"{self.url}/set/{key}",
                headers=self._headers(),
                content=serialized,
                timeout=2.0
            )
            # Set TTL separately.
            httpx.post(
                f"{self.url}/expire/{key}/{ttl_seconds}",
                headers=self._headers(),
                timeout=2.0
            )
            return r.status_code == 200
        except Exception as e:
            logger.debug(f"Cache SET failed for {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete a cache key."""
        if not self._enabled:
            return False
        try:
            httpx.post(
                f"{self.url}/del/{key}",
                headers=self._headers(),
                timeout=2.0
            )
            return True
        except Exception as e:
            logger.debug(f"Cache DEL failed for {key}: {e}")
            return False


# Singleton
_cache = RedisCache()


def get_cache() -> RedisCache:
    return _cache


# ============================================================================
# Cache Key Builder
# ============================================================================

def make_cache_key(prefix: str, **kwargs) -> str:
    """
    Create a deterministic cache key.

    Example:
        key = make_cache_key("search", query="churches tbilisi", top_k=10)
        # → "search:a3f2b1c4d5e6..."
    """
    # Sort kwargs for deterministic keys.
    content = json.dumps(kwargs, sort_keys=True, ensure_ascii=False)
    hash_suffix = hashlib.md5(content.encode()).hexdigest()[:12]
    return f"{prefix}:{hash_suffix}"


# ============================================================================
# TTL Constants
# ============================================================================

SEARCH_TTL = 3600
DISTANCE_TTL = 86400
WEATHER_TTL = 10800


# ============================================================================
# Tool Retry Policy
# ============================================================================

def _is_tool_retryable(exc: Exception) -> bool:
    """Return whether a tool exception is retryable."""
    msg = str(exc).lower()
    return (
        "timeout" in msg
        or "connection" in msg
        or "503" in msg
        or "502" in msg
        or "429" in msg
        or isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))
    )


def with_retry(max_retries: int = 2, base_delay: float = 0.5):
    """
    Retry decorator for sync functions.

    Args:
        max_retries: Maximum retry attempts.
        base_delay: Base delay, doubled after each failed attempt.

    Example:
        @with_retry(max_retries=2, base_delay=0.5)
        def my_tool(query: str) -> dict:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if not _is_tool_retryable(e) or attempt == max_retries:
                        raise
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"Tool {func.__name__} failed (attempt {attempt+1}/{max_retries+1}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    import time
                    time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator


def with_async_retry(max_retries: int = 2, base_delay: float = 0.5):
    """
    Retry decorator for async functions.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if not _is_tool_retryable(e) or attempt == max_retries:
                        raise
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"Tool {func.__name__} failed (attempt {attempt+1}/{max_retries+1}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    await asyncio.sleep(delay)
            raise last_exc
        return wrapper
    return decorator


# ============================================================================
# Cached Tool Wrappers
# ============================================================================

def cached_search_qdrant(query: str, top_k: int = 10, filters: Optional[dict] = None) -> list:
    """
    search_qdrant wrapper with Redis cache and retry.

    Cache hits return immediately; misses call Qdrant and store the result.
    """
    cache = get_cache()
    cache_key = make_cache_key("search", query=query, top_k=top_k, filters=filters)

    # Cache hit
    cached = cache.get(cache_key)
    if cached is not None:
        logger.info(f"Cache HIT: search '{query[:40]}' ({len(cached)} results)")
        return cached

    # Cache miss: real search with retry.
    from src.tools.search_tools import search_qdrant

    @with_retry(max_retries=2, base_delay=0.5)
    def _search():
        return search_qdrant.invoke({"query": query, "top_k": top_k, "filters": filters})

    try:
        results = _search()
        cache.set(cache_key, results, ttl_seconds=SEARCH_TTL)
        logger.info(f"Cache MISS+SET: search '{query[:40]}' ({len(results)} results)")
        return results
    except Exception as e:
        logger.error(f"search_qdrant failed after retries: {e}")
        return []


def cached_get_route(
    start_lon: float, start_lat: float,
    end_lon: float, end_lat: float
) -> dict:
    """
    get_route wrapper with Redis cache and retry.
    """
    cache = get_cache()
    # Round coordinates for stable keys.
    cache_key = make_cache_key(
        "distance",
        slon=round(start_lon, 4), slat=round(start_lat, 4),
        elon=round(end_lon, 4), elat=round(end_lat, 4)
    )

    cached = cache.get(cache_key)
    if cached is not None:
        logger.info(f"Cache HIT: route ({start_lat:.3f},{start_lon:.3f})→({end_lat:.3f},{end_lon:.3f})")
        return cached

    from src.tools.geo_tools import get_route

    @with_retry(max_retries=2, base_delay=1.0)
    def _route():
        return get_route.invoke({
            "start_lon": start_lon, "start_lat": start_lat,
            "end_lon": end_lon, "end_lat": end_lat
        })

    try:
        result = _route()
        if "error" not in result:
            cache.set(cache_key, result, ttl_seconds=DISTANCE_TTL)
            logger.info(f"Cache MISS+SET: route {result.get('distance_km')}km")
        return result
    except Exception as e:
        logger.error(f"get_route failed after retries: {e}")
        return {"error": str(e)}


def cached_geocode_city(city: str) -> dict:
    """
    geocode_city wrapper with Redis cache and retry.
    """
    cache = get_cache()
    cache_key = make_cache_key("geocode", city=city.lower().strip())

    cached = cache.get(cache_key)
    if cached is not None:
        logger.info(f"Cache HIT: geocode '{city}'")
        return cached

    from src.tools.geo_tools import geocode_city

    @with_retry(max_retries=2, base_delay=0.5)
    def _geocode():
        return geocode_city.invoke({"city": city})

    try:
        result = _geocode()
        if "error" not in result:
            cache.set(cache_key, result, ttl_seconds=DISTANCE_TTL)
            logger.info(f"Cache MISS+SET: geocode '{city}' → {result.get('lat')},{result.get('lon')}")
        return result
    except Exception as e:
        logger.error(f"geocode_city failed after retries: {e}")
        return {"error": str(e)}


def cached_get_weather(location: str) -> dict:
    """
    get_weather wrapper with Redis cache and retry.
    """
    cache = get_cache()
    cache_key = make_cache_key("weather", location=location.lower().strip())

    cached = cache.get(cache_key)
    if cached is not None:
        logger.info(f"Cache HIT: weather '{location}'")
        return cached

    from src.tools.weather_tools import get_weather

    @with_retry(max_retries=2, base_delay=0.5)
    def _weather():
        return get_weather.invoke({"location": location})

    try:
        result = _weather()
        if "error" not in result:
            cache.set(cache_key, result, ttl_seconds=WEATHER_TTL)
        return result
    except Exception as e:
        logger.error(f"get_weather failed after retries: {e}")
        return {"error": str(e)}
