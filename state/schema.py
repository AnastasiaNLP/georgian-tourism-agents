from __future__ import annotations

import operator
from typing import Any, Annotated, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from state.contracts import (
    CanonicalTravelState,
    MemorySnapshot,
    QueryClassification,
    TripParameters,
    ValidationResult,
    build_canonical_state,
)


class TravelPlanningState(TypedDict, total=False):
    request_id: str
    user_query: str
    user_language: str
    user_id: str
    correlation_id: str

    intent: str
    conversation_stage: str
    has_current_plan: bool
    current_plan_status: str
    current_plan_version: int
    current_plan: dict[str, Any]
    feedback: str
    follow_up_questions: list[str]

    trip_parameters: TripParameters | dict[str, Any]
    user_memory: MemorySnapshot | dict[str, Any]

    search_results: list[dict[str, Any]]
    enriched_places: list[dict[str, Any]]
    distance_matrix: dict[str, dict[str, Any]]
    raw_itinerary: dict[str, Any]
    enriched_itinerary: dict[str, Any]
    validation_result: ValidationResult | dict[str, Any]
    final_response: str

    feature_flags: dict[str, bool]
    budget_state: dict[str, Any]
    agent_history: Annotated[list[str], operator.add]
    agent_scratchpad: dict[str, Any]
    router_trace: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]
    messages: Annotated[list[AnyMessage], add_messages]

    orchestration: QueryClassification | dict[str, Any] | None
    orchestrator_plan: dict[str, Any]
    orchestrator_decision: str
    orchestrator_params: dict[str, Any]

    execution_start_time: float
    execution_mode: str
    memory_saved: bool
    eval_score: float
    task_complete: bool


def _strip_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_none(inner) for key, inner in value.items() if inner is not None}
    if isinstance(value, list):
        return [_strip_none(item) for item in value if item is not None]
    return value


def create_initial_state(
    user_query: str,
    request_id: str,
    *,
    correlation_id: Optional[str] = None,
    user_id: str = "",
    user_language: str = "en",
) -> TravelPlanningState:
    state = build_canonical_state(
        user_query=user_query,
        request_id=request_id,
        correlation_id=correlation_id or request_id,
        user_id=user_id,
        user_language=user_language,
    )
    runtime_state = state.model_dump(exclude_none=True)
    runtime_state.setdefault("trip_parameters", {})
    runtime_state.setdefault("user_memory", {})
    runtime_state.setdefault("feature_flags", {})
    runtime_state.setdefault("budget_state", {})
    runtime_state.setdefault("agent_history", [])
    runtime_state.setdefault("agent_scratchpad", {})
    runtime_state.setdefault("router_trace", [])
    runtime_state.setdefault("errors", [])
    runtime_state.setdefault("follow_up_questions", [])
    runtime_state.setdefault("search_results", [])
    runtime_state.setdefault("enriched_places", [])
    runtime_state.setdefault("distance_matrix", {})
    runtime_state.setdefault("raw_itinerary", {})
    runtime_state.setdefault("enriched_itinerary", {})
    runtime_state.setdefault("validation_result", {})
    runtime_state.setdefault("final_response", "")
    runtime_state.setdefault("messages", [])
    runtime_state.setdefault("orchestration", None)
    runtime_state.setdefault("orchestrator_plan", {})
    runtime_state.setdefault("orchestrator_decision", "")
    runtime_state.setdefault("orchestrator_params", {})
    runtime_state.setdefault("execution_mode", "api")
    runtime_state.setdefault("memory_saved", False)
    runtime_state.setdefault("eval_score", 0.0)
    runtime_state.setdefault("task_complete", False)
    runtime_state.setdefault("conversation_stage", "CONSULT")
    runtime_state.setdefault("has_current_plan", False)
    runtime_state.setdefault("current_plan_status", "")
    runtime_state.setdefault("current_plan_version", 0)
    runtime_state.setdefault("current_plan", {})
    runtime_state.setdefault("feedback", "")
    runtime_state.setdefault("intent", "INFO")
    runtime_state.setdefault("user_query", user_query)
    runtime_state.setdefault("request_id", request_id)
    runtime_state.setdefault("user_id", user_id)
    runtime_state.setdefault("correlation_id", correlation_id or request_id)
    runtime_state.setdefault("user_language", user_language or "en")
    runtime_state["messages"] = []
    return runtime_state  # type: ignore[return-value]


def to_canonical_state(state: dict[str, Any]) -> CanonicalTravelState:
    cleaned = _strip_none(state)
    cleaned.setdefault("trip_parameters", {})
    cleaned.setdefault("user_memory", {})
    cleaned.setdefault("validation_result", {})
    cleaned.setdefault("distance_matrix", {})
    cleaned.setdefault("follow_up_questions", [])
    cleaned.setdefault("orchestration", None)
    return CanonicalTravelState.model_validate(cleaned)

