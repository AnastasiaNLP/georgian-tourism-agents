"""
LangGraph state schema using TypedDict with annotated reducers.

TypedDict is used because LangGraph nodes return partial updates and merge
state through reducers.

Reducer rules:
- operator.add: append-only fields
- add_messages: message merge with id-based deduplication
- all other fields: last update wins
"""

import operator
from typing import TypedDict, Annotated, Optional
from datetime import datetime
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class TravelPlanningState(TypedDict):
    """
    Main state for the LangGraph workflow.

    Each node returns a dict with only the fields it updates.
    """

    # ── Request metadata ─────────────────────────────────────────────────────
    request_id: str
    correlation_id: str
    user_id: Optional[str]
    user_query: str
    user_language: str

    # ── Orchestrator messages (not agent internals) ─────────────────────────
    messages: Annotated[list[AnyMessage], add_messages]

    # ── Orchestrator control ──────────────────────────────────────────────────
    orchestrator_plan: Optional[dict]       # {steps: [{agent, params}], reasoning}
    orchestrator_decision: Optional[str]    # next agent name
    orchestrator_params: Optional[dict]     # next agent params
    task_complete: bool

    # ── Intent & Trip ─────────────────────────────────────────────────────────
    intent: Optional[str]                   # "PLAN" | "SEARCH" | "INFORMATION"
    query_type: Optional[str]
    trip_parameters: Optional[dict]

    # ── Agent outputs (overwrite semantics) ─────────────────────────────────
    search_results: Optional[list[dict]]
    search_context: Optional[dict]          # {region, filters, center_location}
    weather_data: Optional[dict]
    enriched_places: Optional[list]
    distance_matrix: Optional[dict]         # {"A→B": {"km": 25, "hours": 0.4}, ...}
    raw_itinerary: Optional[dict]
    enriched_itinerary: Optional[dict]
    geocoded_locations: Optional[dict]
    validation_result: Optional[dict]
    final_response: Optional[str]

    # ── Agent scratchpad (overwritten by each agent) ────────────────────────
    agent_scratchpad: Optional[dict]

    # ── Memory ────────────────────────────────────────────────────────────────
    user_memory: Optional[dict]
    relevant_episodes: list[dict]
    eval_score: Optional[dict]
    memory_saved: bool

    # ── Execution control ─────────────────────────────────────────────────────
    execution_mode: str                     # "normal" | "degraded" | "emergency"
    execution_start_time: Optional[datetime]
    feature_flags: dict

    # ── Tracking — append-only through operator.add ─────────────────────────
    # Nodes must return lists for these fields.
    agent_history: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    router_trace: Annotated[list[dict], operator.add]

    # ── Budget tracking (overwrite) ─────────────────────────────────────────
    budget_state: dict


def _detect_language(text: str) -> str:
    """Infer language from Cyrillic, Georgian, and Latin character ratios."""
    cyrillic = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
    georgian = sum(1 for c in text if '\u10D0' <= c <= '\u10FF')
    latin    = sum(1 for c in text if 'a' <= c.lower() <= 'z')
    total = cyrillic + georgian + latin
    if total == 0:
        return "en"
    if cyrillic / total > 0.3:
        return "ru"
    if georgian / total > 0.3:
        return "ka"
    return "en"


def create_initial_state(
    user_query: str,
    request_id: str,
    correlation_id: str,
    user_id: Optional[str] = None,
    user_language: Optional[str] = None,
    trip_parameters: Optional[dict] = None,
) -> TravelPlanningState:
    # Infer language from the query if it was not provided.
    if not user_language:
        user_language = _detect_language(user_query)
    """
    Create the initial state for a new request.

    Optional fields start as None, list fields as [], and dict fields as {}.
    """
    from langchain_core.messages import HumanMessage
    from datetime import timezone

    return TravelPlanningState(
        # Request
        request_id=request_id,
        correlation_id=correlation_id,
        user_id=user_id,
        user_query=user_query,
        user_language=user_language,

        # Start with the original user query.
        messages=[HumanMessage(content=user_query)],

        # Orchestrator fields are filled during graph execution.
        orchestrator_plan=None,
        orchestrator_decision=None,
        orchestrator_params=None,
        task_complete=False,

        # Intent
        intent=None,
        query_type=None,
        trip_parameters=trip_parameters,

        # Agent outputs are empty until their nodes run.
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

        # Memory
        user_memory=None,
        relevant_episodes=[],
        eval_score=None,
        memory_saved=False,

        # Execution
        execution_mode="normal",
        execution_start_time=datetime.now(timezone.utc),
        feature_flags={},

        # Tracking fields are append-only through reducers.
        agent_history=[],
        errors=[],
        router_trace=[],

        # Budget
        budget_state={
            "llm_calls": 0,
            "tool_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "estimated_cost_usd": 0.0,
        },
    )
