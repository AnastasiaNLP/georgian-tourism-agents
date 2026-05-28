"""Runtime state for the LangGraph workflow.

TravelPlanningState is a TypedDict — LangGraph requires it to resolve
Annotated reducers at graph-compile time. Pydantic models live in
state/contracts.py and are used for validation at the boundary.

Reducer rules:
- operator.add  : append-only (agent_history, errors, router_trace)
- add_messages  : message merge with id-based deduplication
- all other     : last-write wins
"""

from __future__ import annotations

import operator
from datetime import datetime, timezone
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

from state.contracts import (
    CanonicalTravelState,
    MemorySnapshot,
    QueryClassification,
    TripParameters,
    ValidationResult,
    build_canonical_state,
)


class TravelPlanningState(TypedDict):
    """LangGraph runtime state.

    Each node returns a partial dict; LangGraph merges it into this state
    using the reducers declared via Annotated.
    """

    # ── Request metadata ─────────────────────────────────────────────────────
    request_id: str
    correlation_id: str
    user_id: Optional[str]
    user_query: str
    user_language: str

    # ── LangGraph messages (dedup reducer) ───────────────────────────────────
    messages: Annotated[list[AnyMessage], add_messages]

    # ── Orchestrator control (legacy bridge, kept until graph migration) ─────
    orchestrator_plan: Optional[dict]
    orchestrator_decision: Optional[str]
    orchestrator_params: Optional[dict]
    task_complete: bool

    # ── Intent and conversation stage ────────────────────────────────────────
    # intent: legacy "PLAN" | "SEARCH" written by orchestrator_plan_node
    # conversation_stage: new canonical "CONSULT"|"PLAN"|"REVISE"|"POST_REVIEW"|"INFO"
    intent: Optional[str]
    conversation_stage: Optional[str]
    query_type: Optional[str]

    # ── Trip parameters and plan tracking ────────────────────────────────────
    trip_parameters: Optional[dict]
    has_current_plan: bool
    current_plan_status: str
    current_plan_version: int
    current_plan: dict
    feedback: str
    follow_up_questions: list[str]

    # ── Agent outputs (last-write wins) ──────────────────────────────────────
    search_results: Optional[list[dict]]
    search_context: Optional[dict]
    weather_data: Optional[dict]
    enriched_places: Optional[list]
    distance_matrix: Optional[dict]
    raw_itinerary: Optional[dict]
    enriched_itinerary: Optional[dict]
    geocoded_locations: Optional[dict]
    validation_result: Optional[dict]
    final_response: Optional[str]
    agent_scratchpad: Optional[dict]
    orchestration: Optional[dict]

    # ── Memory ────────────────────────────────────────────────────────────────
    user_memory: Optional[dict]
    relevant_episodes: list[dict]
    eval_score: Optional[dict]
    memory_saved: bool

    # ── Execution control ─────────────────────────────────────────────────────
    execution_mode: str
    execution_start_time: Optional[datetime]
    feature_flags: dict

    # ── Append-only fields — nodes MUST return lists for these ───────────────
    agent_history: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    router_trace: Annotated[list[dict], operator.add]

    # ── Budget tracking (last-write wins) ────────────────────────────────────
    budget_state: dict


# ── Helpers ──────────────────────────────────────────────────────────────────

def _detect_language(text: str) -> str:
    """Infer language from Cyrillic, Georgian, and Latin character ratios."""
    cyrillic = sum(1 for c in text if "Ѐ" <= c <= "ӿ")
    georgian = sum(1 for c in text if "ა" <= c <= "ჿ")
    latin    = sum(1 for c in text if "a" <= c.lower() <= "z")
    total = cyrillic + georgian + latin
    if total == 0:
        return "en"
    if cyrillic / total > 0.3:
        return "ru"
    if georgian / total > 0.3:
        return "ka"
    return "en"


def create_initial_state(
    *,
    user_query: str,
    request_id: str,
    correlation_id: str,
    user_id: Optional[str] = None,
    user_language: Optional[str] = None,
) -> TravelPlanningState:
    """Build the initial LangGraph state for a new request.

    Uses build_canonical_state as the authoritative factory and maps the
    result into the TypedDict that LangGraph requires at runtime.
    """
    language = user_language or _detect_language(user_query)

    canonical = build_canonical_state(
        request_id=request_id,
        user_query=user_query,
        user_language=language,
        user_id=user_id,
        correlation_id=correlation_id,
    )

    return TravelPlanningState(
        # ── Request ────────────────────────────────────────────────────────
        request_id=canonical.request_id,
        correlation_id=canonical.correlation_id,
        user_id=canonical.user_id,
        user_query=canonical.user_query,
        user_language=canonical.user_language,
        # ── Messages ───────────────────────────────────────────────────────
        messages=[HumanMessage(content=user_query)],
        # ── Orchestrator (legacy bridge) ───────────────────────────────────
        orchestrator_plan=None,
        orchestrator_decision=None,
        orchestrator_params=None,
        task_complete=False,
        # ── Intent / stage ─────────────────────────────────────────────────
        intent=None,
        conversation_stage=canonical.conversation_stage,
        query_type=None,
        # ── Trip ───────────────────────────────────────────────────────────
        trip_parameters=canonical.trip_parameters.model_dump(),
        has_current_plan=False,
        current_plan_status="none",
        current_plan_version=0,
        current_plan={},
        feedback="",
        follow_up_questions=[],
        # ── Agent outputs ──────────────────────────────────────────────────
        search_results=None,
        search_context=None,
        weather_data=None,
        enriched_places=None,
        distance_matrix=None,
        raw_itinerary=None,
        enriched_itinerary=None,
        geocoded_locations=None,
        validation_result=None,
        final_response=None,
        agent_scratchpad=None,
        orchestration=None,
        # ── Memory ─────────────────────────────────────────────────────────
        user_memory=None,
        relevant_episodes=[],
        eval_score=None,
        memory_saved=False,
        # ── Execution ──────────────────────────────────────────────────────
        execution_mode="normal",
        execution_start_time=datetime.now(timezone.utc),
        feature_flags={},
        # ── Append-only (reducer lists — always start empty) ───────────────
        agent_history=[],
        errors=[],
        router_trace=[],
        # ── Budget ─────────────────────────────────────────────────────────
        budget_state={
            "llm_calls": 0,
            "tool_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "estimated_cost_usd": 0.0,
        },
    )


def to_canonical_state(state: dict[str, Any]) -> CanonicalTravelState:
    """Convert a runtime dict into the canonical Pydantic model for validation.

    None values are dropped so Pydantic uses its own field defaults for
    fields that have not been populated yet (e.g. search_results=None
    at graph start becomes [] in the canonical model).
    """
    clean = {k: v for k, v in state.items() if v is not None}
    return CanonicalTravelState.model_validate(clean)


__all__ = [
    "CanonicalTravelState",
    "MemorySnapshot",
    "QueryClassification",
    "TravelPlanningState",
    "TripParameters",
    "ValidationResult",
    "build_canonical_state",
    "create_initial_state",
    "to_canonical_state",
]
