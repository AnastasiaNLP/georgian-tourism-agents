"""Response schemas returned by the API."""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from enum import Enum

from schemas.common import ConfidenceScore
from schemas.itinerary import EnrichedItineraryV1


# ============================================================================
# Degradation Flag
# ============================================================================

class DegradationFlag(str, Enum):
    """Degradation event types."""
    # Geo issues
    APPROXIMATE_ROUTES = "APPROXIMATE_ROUTES"         # ORS API failed, used LLM fallback
    GEOCODING_FAILED = "GEOCODING_FAILED"             # Couldn't geocode location

    # Search issues
    LIMITED_SEARCH_RESULTS = "LIMITED_SEARCH_RESULTS" # Found < N results
    SEARCH_FAILED = "SEARCH_FAILED"                   # Search completely failed

    # Planning issues
    SIMPLIFIED_ITINERARY = "SIMPLIFIED_ITINERARY"     # Reduced activities due to constraints
    PLANNING_RETRY = "PLANNING_RETRY"                 # Had to retry planning

    # API issues
    WEATHER_UNAVAILABLE = "WEATHER_UNAVAILABLE"       # Weather API failed
    AVAILABILITY_UNVERIFIED = "AVAILABILITY_UNVERIFIED" # Couldn't verify availability

    # Budget issues
    BUDGET_WARNING = "BUDGET_WARNING"                 # Approaching budget limits
    TIMEOUT_WARNING = "TIMEOUT_WARNING"               # Approaching time limits


class DegradationFlagV1(BaseModel):
    """One degradation event surfaced to API consumers"""
    source:  str              = Field(..., description="Which agent/component raised this")
    flag:    DegradationFlag  = Field(..., description="Type of degradation")
    message: str              = Field(..., description="Human-readable explanation")

    class Config:
        frozen = True


# ============================================================================
# Confidence
# ============================================================================

class ConfidenceV1(BaseModel):
    """
    Multi-dimensional confidence tracking.

    Tracks confidence per-domain + degradation flags.
    """
    # Per-domain confidence
    route_confidence:        float = Field(1.0, ge=0.0, le=1.0)
    weather_confidence:      float = Field(1.0, ge=0.0, le=1.0)
    availability_confidence: float = Field(1.0, ge=0.0, le=1.0)
    planning_confidence:     float = Field(1.0, ge=0.0, le=1.0)

    # Degradation flags
    approximate_route:       bool = False
    weather_missing:         bool = False
    availability_unverified: bool = False

    # Overall confidence (computed)
    overall_confidence:      float = Field(1.0, ge=0.0, le=1.0)

    class Config:
        frozen = True

    @classmethod
    def build(cls, **kwargs) -> "ConfidenceV1":
        """
        Build confidence with auto-computed overall.

        overall_confidence = min(all domain confidences)
        """
        # Extract confidence values
        route_conf = kwargs.get("route_confidence", 1.0)
        weather_conf = kwargs.get("weather_confidence", 1.0)
        avail_conf = kwargs.get("availability_confidence", 1.0)
        plan_conf = kwargs.get("planning_confidence", 1.0)

        # Compute overall as minimum
        overall = round(min(route_conf, weather_conf, avail_conf, plan_conf), 2)

        # Create with computed overall
        return cls(**{**kwargs, "overall_confidence": overall})


# ============================================================================
# Response Metadata
# ============================================================================

class ResponseMetadata(BaseModel):
    """Metadata about the response"""
    request_id: str
    correlation_id: str
    generated_at: str  # ISO timestamp
    processing_time_ms: float
    agents_used: List[str] = Field(default_factory=list)
    iteration_count: int = 0

    class Config:
        frozen = True


# ============================================================================
# Final Response
# ============================================================================

class FinalResponseV1(BaseModel):
    """
    Everything returned after successful workflow.

    Final API response returned to the client.
    """
    # Main content
    itinerary: Optional[EnrichedItineraryV1] = Field(None, description="Generated itinerary (if applicable)")
    answer: str = Field(..., min_length=1, description="Text answer to user query")
    language: str = Field("en", description="Response language")

    # Quality tracking
    confidence: ConfidenceV1 = Field(default_factory=ConfidenceV1)
    is_degraded: bool = Field(False, description="Was system degraded during execution?")
    degradation_flags: List[DegradationFlagV1] = Field(default_factory=list)

    # Metadata
    metadata: ResponseMetadata
    warnings: List[str] = Field(default_factory=list)

    class Config:
        frozen = True

    @property
    def is_high_confidence(self) -> bool:
        """Is overall confidence high (>= 0.8)?"""
        return self.confidence.overall_confidence >= 0.8

    @property
    def total_days(self) -> int:
        """Number of days in itinerary (0 if no itinerary)"""
        return self.itinerary.total_days if self.itinerary else 0
