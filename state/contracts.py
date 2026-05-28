from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


ConversationStage = Literal["CONSULT", "PLAN", "REVISE", "POST_REVIEW", "INFO"]
ClassifierIntent = Literal["CONSULT", "PLAN", "REVISE", "INFO"]
TravelPace = Literal["relaxed", "moderate", "intensive"]
RecommendedAction = Literal["proceed", "retry", "degrade", "abort"]


class QueryClassification(BaseModel):
    intent: ClassifierIntent = "INFO"
    region: str = ""
    search_query: str = ""
    days: Optional[int] = None
    pace: TravelPace = "moderate"
    should_ask_clarification: bool = False
    conversation_stage: ConversationStage = "CONSULT"
    steps: list[dict[str, Any]] = Field(default_factory=list)
    reasoning: str = ""
    estimated_agents: int = 0


class TripParameters(BaseModel):
    region: str = ""
    days: Optional[int] = None
    pace: TravelPace = "moderate"
    interests: list[str] = Field(default_factory=list)
    budget_level: str = ""
    transport_mode: str = ""
    search_query: str = ""


class MemorySnapshot(BaseModel):
    user_id: str = ""
    preferences: list[str] = Field(default_factory=list)
    visited_places: list[str] = Field(default_factory=list)
    last_language: str = ""
    episode_summaries: list[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    is_valid: bool = False
    score: float = 0.0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    recommended_action: RecommendedAction = "proceed"


class Coordinates(BaseModel):
    lat: Optional[float] = None
    lon: Optional[float] = None


class Activity(BaseModel):
    name: str
    location: str = ""
    category: str = ""
    coordinates: Coordinates = Field(default_factory=Coordinates)
    description: str = ""


class ItineraryDay(BaseModel):
    day: int
    location: str = ""
    activities: list[Activity] = Field(default_factory=list)
    total_distance_km: float = 0.0
    total_driving_hours: float = 0.0


class Itinerary(BaseModel):
    days: list[ItineraryDay] = Field(default_factory=list)
    total_days: int = 0
    pace: TravelPace = "moderate"
    summary: str = ""


class CanonicalTravelState(BaseModel):
    request_id: str = ""
    user_query: str = ""
    user_language: str = "en"
    user_id: str = ""
    correlation_id: str = ""

    intent: ClassifierIntent = "INFO"
    conversation_stage: ConversationStage = "CONSULT"
    has_current_plan: bool = False
    current_plan_status: str = ""
    current_plan_version: int = 0
    current_plan: dict[str, Any] = Field(default_factory=dict)
    feedback: str = ""
    follow_up_questions: list[str] = Field(default_factory=list)

    trip_parameters: TripParameters = Field(default_factory=TripParameters)
    user_memory: MemorySnapshot = Field(default_factory=MemorySnapshot)

    search_results: list[dict[str, Any]] = Field(default_factory=list)
    enriched_places: list[dict[str, Any]] = Field(default_factory=list)
    distance_matrix: dict[str, dict[str, Any]] = Field(default_factory=dict)
    raw_itinerary: dict[str, Any] = Field(default_factory=dict)
    enriched_itinerary: dict[str, Any] = Field(default_factory=dict)
    validation_result: ValidationResult = Field(default_factory=ValidationResult)
    final_response: str = ""

    feature_flags: dict[str, bool] = Field(default_factory=dict)
    budget_state: dict[str, Any] = Field(default_factory=dict)
    agent_history: list[str] = Field(default_factory=list)
    agent_scratchpad: dict[str, Any] = Field(default_factory=dict)
    router_trace: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    orchestration: Optional[Union[QueryClassification, dict[str, Any]]] = None
    task_complete: bool = False


def build_canonical_state(
    *,
    user_query: str,
    request_id: str,
    correlation_id: str,
    user_id: str = "",
    user_language: str = "en",
) -> CanonicalTravelState:
    return CanonicalTravelState(
        request_id=request_id,
        user_query=user_query,
        user_language=user_language or "en",
        user_id=user_id,
        correlation_id=correlation_id,
        intent="INFO",
        conversation_stage="CONSULT",
        has_current_plan=False,
        current_plan_status="",
        current_plan_version=0,
        current_plan={},
        feedback="",
        follow_up_questions=[],
        trip_parameters=TripParameters(),
        user_memory=MemorySnapshot(user_id=user_id),
        search_results=[],
        enriched_places=[],
        distance_matrix={},
        raw_itinerary={},
        enriched_itinerary={},
        validation_result=ValidationResult(),
        final_response="",
        feature_flags={},
        budget_state={},
        agent_history=[],
        agent_scratchpad={},
        router_trace=[],
        errors=[],
        orchestration=None,
        task_complete=False,
    )

