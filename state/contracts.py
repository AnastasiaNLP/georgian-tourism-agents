"""Pydantic contracts for the canonical travel-assistant state.

These models define the public structure of the graph state and the
classifier output. The graph can still serialize to dict at runtime, but
the contract itself stays typed and explicit.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


ConversationStage = Literal["CONSULT", "PLAN", "REVISE", "POST_REVIEW", "INFO"]
ClassifierIntent = Literal["CONSULT", "PLAN", "REVISE", "INFO"]
TravelPace = Literal["relaxed", "moderate", "intensive"]


class QueryClassification(BaseModel):
    """Output produced by the classifier before deterministic routing."""

    intent: ClassifierIntent = Field(description="High-level routing intent")
    region: str = Field(default="", description="Detected Georgian region")
    search_query: str = Field(default="", description="Normalized search query")
    days: Optional[int] = Field(default=None, ge=1, le=30)
    pace: TravelPace = Field(default="moderate")
    should_ask_clarification: bool = Field(
        default=False,
        description="True when the system should ask for more details before planning",
    )
    steps: list[dict] = Field(default_factory=list)
    reasoning: str = ""
    estimated_agents: int = 0


class TripParameters(BaseModel):
    """Normalized trip parameters shared across planning stages."""

    region: str = ""
    days: Optional[int] = Field(default=None, ge=1, le=30)
    pace: TravelPace = "moderate"
    search_query: str = ""
    consent_to_plan: bool = False


class MemorySnapshot(BaseModel):
    """User memory visible to the classifier and planning stages."""

    visited_places: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    preferred_language: str = "en"
    disliked_places: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """Structured validation result for itinerary checks."""

    is_valid: bool = True
    score: float = 1.0
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    recommended_action: Literal["proceed", "retry", "degrade", "abort"] = "proceed"


class CanonicalTravelState(BaseModel):
    """Canonical graph state shared by the parent graph and subgraphs."""

    request_id: str
    user_query: str
    user_language: str = "en"
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None

    # Routing and dialogue control
    intent: Optional[ClassifierIntent] = None
    conversation_stage: Optional[ConversationStage] = None
    has_current_plan: bool = False
    current_plan_status: str = "none"
    current_plan_version: int = 0
    current_plan: dict = Field(default_factory=dict)
    feedback: str = ""
    follow_up_questions: list[str] = Field(default_factory=list)

    # Inputs and memory
    trip_parameters: TripParameters = Field(default_factory=TripParameters)
    user_memory: Optional[MemorySnapshot] = None

    # Working data produced by agents
    search_results: list[dict] = Field(default_factory=list)
    enriched_places: list[dict] = Field(default_factory=list)
    distance_matrix: dict = Field(default_factory=dict)
    raw_itinerary: dict = Field(default_factory=dict)
    enriched_itinerary: dict = Field(default_factory=dict)
    validation_result: Optional[ValidationResult] = None
    final_response: str = ""

    # Control and observability
    feature_flags: dict = Field(default_factory=dict)
    budget_state: dict = Field(default_factory=dict)
    agent_history: list[str] = Field(default_factory=list)
    agent_scratchpad: dict = Field(default_factory=dict)
    router_trace: list[dict] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    orchestration: Optional[QueryClassification] = None


def build_canonical_state(
    *,
    request_id: str,
    user_query: str,
    user_language: str = "en",
    user_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> CanonicalTravelState:
    """Create the canonical initial graph state."""

    return CanonicalTravelState(
        request_id=request_id,
        user_query=user_query,
        user_language=user_language,
        user_id=user_id,
        correlation_id=correlation_id or request_id,
        conversation_stage="CONSULT",
        user_memory=None,
        trip_parameters=TripParameters(),
        feature_flags={},
        budget_state={},
        agent_history=[],
        agent_scratchpad={},
        router_trace=[],
        errors=[],
        current_plan={},
        raw_itinerary={},
        enriched_itinerary={},
        search_results=[],
        enriched_places=[],
        distance_matrix={},
        follow_up_questions=[],
        final_response="",
        has_current_plan=False,
        current_plan_status="none",
        current_plan_version=0,
        feedback="",
        orchestration=None,
    )

