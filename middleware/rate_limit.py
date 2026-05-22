"""In-memory rate limiter."""
import time
import logging
from collections import defaultdict
from fastapi import Request
from fastapi.responses import JSONResponse
from config.settings import get_settings

logger = logging.getLogger(__name__)

_requests: dict = defaultdict(list)
MAX_RPM = get_settings().rate_limit_rpm


async def rate_limit_middleware(request: Request, call_next):
    """Allow at most MAX_RPM requests per minute from one IP."""
    # Skip public utility endpoints.
    if request.url.path in ("/api/v1/health", "/metrics", "/docs", "/openapi.json", "/"):
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    # Drop entries older than 60 seconds.
    _requests[client_ip] = [t for t in _requests[client_ip] if now - t < 60]

    if len(_requests[client_ip]) >= MAX_RPM:
        logger.warning(f"Rate limit exceeded for {client_ip}")
        return JSONResponse(
            status_code=429,
            content={"detail": f"Rate limit exceeded. Max {MAX_RPM} requests per minute."},
        )

    _requests[client_ip].append(now)
    return await call_next(request)
