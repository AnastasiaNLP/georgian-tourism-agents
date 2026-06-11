"""
Retry and cache helpers for external tools.

TTL strategy:
- search cache: 1 hour
- distance cache: 24 hours
- weather cache: 3 hours
"""

import asyncio
import json
import hashlib
import logging
import urllib.parse
import uuid
from typing import Any, Optional, Callable
from functools import wraps
import httpx
from config.settings import get_settings
from src.tools.geo_tools import GEO_FOCUS_LAT, GEO_FOCUS_LON

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Upstash Redis over HTTP API — fully async.

    Uses a single AsyncClient per cache instance (lazy-created on first use).
    Call aclose() during application shutdown to release the connection pool.
    """

    def __init__(self):
        settings = get_settings()
        self.url = settings.upstash_redis_url or ""
        self.token = settings.upstash_redis_token or ""
        self._enabled = bool(self.url and self.token)
        self._http_client: Optional[httpx.AsyncClient] = None

        if not self._enabled:
            logger.warning("Redis cache disabled — UPSTASH_REDIS_URL or TOKEN not set")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}

    def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=2.0)
        return self._http_client

    async def aclose(self) -> None:
        """Close the underlying HTTP client. Call once during app shutdown."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache. Returns None when missing or on error."""
        if not self._enabled:
            return None
        try:
            r = await self._get_client().get(
                f"{self.url}/get/{key}",
                headers=self._headers(),
            )
            data = r.json()
            result = data.get("result")
            if result is None:
                return None
            return json.loads(result)
        except Exception as e:
            logger.debug(f"Cache GET failed for {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Save a value with TTL — single atomic request using EX query param."""
        if not self._enabled:
            return False
        try:
            serialized = json.dumps(value, ensure_ascii=False)
            r = await self._get_client().post(
                f"{self.url}/set/{key}",
                headers=self._headers(),
                content=serialized,
                params={"EX": ttl_seconds},
            )
            return r.status_code == 200
        except Exception as e:
            logger.debug(f"Cache SET failed for {key}: {e}")
            return False

    async def acquire_lock(self, key: str, ttl_seconds: int = 360) -> Optional[str]:
        """SET key <token> NX EX ttl — returns unique token if acquired, None if lock is held."""
        if not self._enabled:
            return None
        token = uuid.uuid4().hex
        try:
            r = await self._get_client().post(
                f"{self.url}/set/{key}",
                headers=self._headers(),
                content=token,
                params={"EX": ttl_seconds, "NX": ""},
            )
            data = r.json()
            return token if data.get("result") == "OK" else None
        except Exception as e:
            logger.debug(f"Cache ACQUIRE_LOCK failed for {key}: {e}")
            return None

    async def release_lock(self, key: str, token: str) -> bool:
        """Compare-and-delete via Lua EVAL: only removes the lock if the stored value matches token."""
        if not self._enabled:
            return False
        # Atomic: if get(key)==token then del(key) — prevents releasing another owner's lock.
        script = "if redis.call('get',KEYS[1])==ARGV[1] then return redis.call('del',KEYS[1]) else return 0 end"
        path = "/".join([
            "eval",
            urllib.parse.quote(script, safe=""),
            "1",
            urllib.parse.quote(key, safe=""),
            urllib.parse.quote(token, safe=""),
        ])
        try:
            r = await self._get_client().post(
                f"{self.url}/{path}",
                headers=self._headers(),
            )
            data = r.json()
            return bool(data.get("result"))
        except Exception as e:
            logger.debug(f"Cache RELEASE_LOCK failed for {key}: {e}")
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
# Tool Retry Policy (sync — safe inside asyncio.to_thread)
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

    Intended to wrap sync tool calls that run inside asyncio.to_thread —
    time.sleep blocks the thread pool thread, not the event loop.

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
                    import time
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"Tool {func.__name__} failed (attempt {attempt+1}/{max_retries+1}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator


# ============================================================================
# Cached Tool Wrappers — all async
# ============================================================================

async def cached_search_qdrant(
    query: str,
    top_k: int = 10,
    filters: Optional[dict] = None,
) -> list:
    """
    search_qdrant wrapper with async Redis cache and sync retry.

    The sync Qdrant call runs in a thread pool via asyncio.to_thread
    so it never blocks the event loop.
    """
    cache = get_cache()
    cache_key = make_cache_key("search", query=query, top_k=top_k, filters=filters)

    cached = await cache.get(cache_key)
    if cached is not None:
        logger.info(f"Cache HIT: search '{query[:40]}' ({len(cached)} results)")
        return cached

    from src.tools.search_tools import search_qdrant

    @with_retry(max_retries=2, base_delay=0.5)
    def _search():
        return search_qdrant.invoke({"query": query, "top_k": top_k, "filters": filters})

    try:
        results = await asyncio.to_thread(_search)
        await cache.set(cache_key, results, ttl_seconds=SEARCH_TTL)
        logger.info(f"Cache MISS+SET: search '{query[:40]}' ({len(results)} results)")
        return results
    except Exception as e:
        logger.error(f"search_qdrant failed after retries: {e}")
        return []


async def cached_get_route(
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
) -> dict:
    """
    get_route wrapper with async Redis cache and sync retry.
    """
    cache = get_cache()
    cache_key = make_cache_key(
        "distance",
        slon=round(start_lon, 4), slat=round(start_lat, 4),
        elon=round(end_lon, 4), elat=round(end_lat, 4),
    )

    cached = await cache.get(cache_key)
    if cached is not None:
        logger.info(
            f"Cache HIT: route ({start_lat:.3f},{start_lon:.3f})"
            f"→({end_lat:.3f},{end_lon:.3f})"
        )
        return cached

    from src.tools.geo_tools import get_route

    @with_retry(max_retries=2, base_delay=1.0)
    def _route():
        return get_route.invoke({
            "start_lon": start_lon, "start_lat": start_lat,
            "end_lon": end_lon, "end_lat": end_lat,
        })

    try:
        result = await asyncio.to_thread(_route)
        if "error" not in result:
            await cache.set(cache_key, result, ttl_seconds=DISTANCE_TTL)
            logger.info(f"Cache MISS+SET: route {result.get('distance_km')}km")
        return result
    except Exception as e:
        logger.error(f"get_route failed after retries: {e}")
        return {"error": str(e)}


async def cached_geocode_city(
    city: str,
    focus_lat: float = GEO_FOCUS_LAT,
    focus_lon: float = GEO_FOCUS_LON,
) -> dict:
    """
    geocode_city wrapper with async Redis cache and sync retry.

    focus_lat/focus_lon are passed to ORS as a proximity hint to reduce
    hallucination for objects far from the Georgia centroid (e.g. Svaneti peaks).
    Rounded to 1 decimal in the cache key — precise enough for regional grouping.
    Trade-off: the same place queried with different focus points creates separate
    cache entries. Correctness > hit-rate here (wrong coords are worse than a miss).
    """
    cache = get_cache()
    cache_key = make_cache_key(
        "geocode",
        city=city.lower().strip(),
        flat=round(focus_lat, 1),
        flon=round(focus_lon, 1),
    )

    cached = await cache.get(cache_key)
    if cached is not None:
        logger.info(f"Cache HIT: geocode '{city}'")
        return cached

    from src.tools.geo_tools import geocode_city

    @with_retry(max_retries=2, base_delay=0.5)
    def _geocode():
        return geocode_city.invoke({"city": city, "focus_lat": focus_lat, "focus_lon": focus_lon})

    try:
        result = await asyncio.to_thread(_geocode)
        if "error" not in result:
            await cache.set(cache_key, result, ttl_seconds=DISTANCE_TTL)
            logger.info(
                f"Cache MISS+SET: geocode '{city}' → {result.get('lat')},{result.get('lon')}"
            )
        return result
    except Exception as e:
        logger.error(f"geocode_city failed after retries: {e}")
        return {"error": str(e)}


async def cached_get_weather(location: str) -> dict:
    """
    get_weather wrapper with async Redis cache and sync retry.
    """
    cache = get_cache()
    cache_key = make_cache_key("weather", location=location.lower().strip())

    cached = await cache.get(cache_key)
    if cached is not None:
        logger.info(f"Cache HIT: weather '{location}'")
        return cached

    from src.tools.weather_tools import get_weather

    @with_retry(max_retries=2, base_delay=0.5)
    def _weather():
        return get_weather.invoke({"location": location})

    try:
        result = await asyncio.to_thread(_weather)
        if "error" not in result:
            await cache.set(cache_key, result, ttl_seconds=WEATHER_TTL)
        return result
    except Exception as e:
        logger.error(f"get_weather failed after retries: {e}")
        return {"error": str(e)}
