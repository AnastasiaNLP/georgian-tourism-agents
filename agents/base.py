"""Shared helpers for graph agent nodes."""

from __future__ import annotations
import logging
from typing import Any, Callable, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def now_iso() -> str:
    """Current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def extract_search_params(state: dict) -> dict:
    """
    Extract search parameters from orchestrator_params and state.
    
    The orchestrator sends params through orchestrator_params, for example
    {"region": "Adjara", "max_results": 10}.
    """
    params = dict(state.get("orchestrator_params") or {})
    
    # Fall back to trip parameters when the orchestrator did not set a region.
    if not params.get("region"):
        trip = state.get("trip_parameters") or {}
        if trip.get("start_location"):
            params["region"] = trip["start_location"]
    
    # Result count.
    if not params.get("max_results"):
        flags = state.get("feature_flags") or {}
        params["max_results"] = flags.get("MAX_SEARCH_RESULTS", 10)
    
    return params


def extract_trip_days(state: dict) -> int:
    """Extract requested trip duration from state."""
    params = state.get("orchestrator_params") or {}
    if params.get("days"):
        return int(params["days"])
    
    trip = state.get("trip_parameters") or {}
    if trip.get("days"):
        return int(trip["days"])
    
    # Fall back to parsing the original query.
    from orchestrator.fallback_rules import _detect_days
    query = state.get("user_query", "").lower()
    return _detect_days(query)


def extract_pace(state: dict) -> str:
    """Extract travel pace from state."""
    params = state.get("orchestrator_params") or {}
    if params.get("pace"):
        return params["pace"]
    
    trip = state.get("trip_parameters") or {}
    return trip.get("pace", "moderate")


def make_scratchpad(agent: str, summary: str, **extra) -> dict:
    """
    Create agent_scratchpad with a short summary.
    
    The scratchpad is useful for diagnostics and optional review steps.
    """
    return {
        "agent": agent,
        "summary": summary,
        "ts": now_iso(),
        **extra,
    }
