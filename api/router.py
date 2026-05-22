"""
API router for LangGraph-backed planning endpoints.
"""
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from middleware.auth import verify_api_key

from monitoring.tracing import get_run_config
from monitoring.metrics import ACTIVE_REQUESTS
from state.schema import create_initial_state
from config.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()

class PlanRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000,
                       description="User's travel query")
    user_id: Optional[str] = Field(None, description="Optional user ID for memory")
    language: str = Field("ru", description="Response language: ru/en/ka")

class PlanResponse(BaseModel):
    request_id: str
    status: str
    response: str
    execution_mode: str
    agent_history: List[str]
    has_itinerary: bool
    eval_score: Optional[dict]
    memory_saved: bool
    duration_seconds: float

_guardrails = None

def _get_guardrails():
    global _guardrails
    if _guardrails is None:
        from guardrails.input_guardrails import InputGuardrails
        _guardrails = InputGuardrails()
    return _guardrails

@router.post("/plan", response_model=PlanResponse, dependencies=[Depends(verify_api_key)])
async def plan_trip(request: PlanRequest, http_request: Request):
    """
    Main planning endpoint.
    Runs the LangGraph orchestrator loop and returns a travel itinerary.
    """
    request_id = str(uuid.uuid4())[:8]
    started_at = datetime.now(timezone.utc)

    logger.info(f"[{request_id}] POST /plan query={request.query[:80]!r} "
                f"lang={request.language} user={request.user_id}")

    guard = _get_guardrails().check(request.query, request.language)
    if not guard.passed:
        logger.warning(f"[{request_id}] Guardrail REJECT: {guard.check_failed}")
        raise HTTPException(status_code=400, detail=guard.reason)

    state = create_initial_state(
        user_query=request.query,
        request_id=request_id,
        correlation_id=str(uuid.uuid4()),
        user_id=request.user_id,
        user_language=request.language,
    )

    graph = getattr(http_request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialized")

    config = get_run_config(
        request_id=request_id,
        user_id=request.user_id,
    )

    ACTIVE_REQUESTS.inc()
    try:
        result = await graph.ainvoke(state, config=config)
    except Exception as e:
        logger.error(f"[{request_id}] Graph failed: {e}")
        raise HTTPException(status_code=500, detail=f"Graph error: {str(e)[:200]}")
    finally:
        ACTIVE_REQUESTS.dec()

    duration = (datetime.now(timezone.utc) - started_at).total_seconds()

    final_response = result.get("final_response") or "No response generated."
    has_itinerary = bool(
        result.get("enriched_itinerary") or result.get("raw_itinerary")
    )

    logger.info(
        f"[{request_id}] Done in {duration:.1f}s "
        f"mode={result.get('execution_mode')} "
        f"agents={result.get('agent_history', [])}"
    )

    return PlanResponse(
        request_id=request_id,
        status="ok",
        response=final_response,
        execution_mode=result.get("execution_mode", "normal"),
        agent_history=result.get("agent_history", []),
        has_itinerary=has_itinerary,
        eval_score=result.get("eval_score"),
        memory_saved=result.get("memory_saved", False),
        duration_seconds=round(duration, 2),
    )


@router.get("/health")
async def health():
    """Health check for runtime dependencies."""
    from fastapi.responses import JSONResponse
    checks = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    overall = "healthy"
    settings = get_settings()

    openai_ok    = bool(settings.openai_api_key)
    anthropic_ok = bool(settings.anthropic_api_key)
    checks["openai_key"]    = "set" if openai_ok    else "MISSING"
    checks["anthropic_key"] = "set" if anthropic_ok else "MISSING"
    if not openai_ok or not anthropic_ok:
        overall = "unhealthy"

    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=5,
        )
        collections = [c.name for c in client.get_collections().collections]
        checks["qdrant"] = f"ok ({len(collections)} collections)"
    except Exception as e:
        checks["qdrant"] = f"error: {str(e)[:80]}"
        if overall == "healthy":
            overall = "degraded"

    try:
        from src.tools.tool_cache import get_cache
        cache = get_cache()
        if not cache.enabled:
            checks["redis"] = "disabled (no credentials)"
        else:
            cache.set("health:ping", "pong", ttl_seconds=10)
            val = cache.get("health:ping")
            checks["redis"] = "ok" if val == "pong" else f"unexpected value: {val}"
    except Exception as e:
        checks["redis"] = f"error: {str(e)[:80]}"
        if overall == "healthy":
            overall = "degraded"

    checks["status"] = overall
    status_code = 503 if overall == "unhealthy" else 200
    return JSONResponse(content=checks, status_code=status_code)
