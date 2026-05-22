"""
Common schemas used across the application.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal
from enum import Enum


# ============================================================================
# Coordinates
# ============================================================================

class Coordinates(BaseModel):
    """
    Geographic coordinates.

    Used for geographic points.
    """
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude (-90 to 90)")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude (-180 to 180)")

    class Config:
        frozen = True  # Immutable


# ============================================================================
# Confidence Score
# ============================================================================

class ConfidenceScore(BaseModel):
    """
    Confidence score with source tracking.

    Typed confidence score with source metadata.
    """
    value: float = Field(..., ge=0.0, le=1.0, description="Confidence value (0.0 to 1.0)")
    source: str = Field(default="unknown", description="Source of confidence (e.g., 'api', 'llm', 'fallback')")

    class Config:
        frozen = True

    @classmethod
    def from_float(cls, value: float, source: str = "unknown") -> "ConfidenceScore":
        """Create ConfidenceScore from float"""
        return cls(value=value, source=source)

    def __float__(self) -> float:
        """Allow using as float"""
        return self.value

    def __eq__(self, other) -> bool:
        """Allow comparison with float"""
        if isinstance(other, (int, float)):
            return self.value == other
        if isinstance(other, ConfidenceScore):
            return self.value == other.value
        return False

    def __lt__(self, other) -> bool:
        if isinstance(other, (int, float)):
            return self.value < other
        if isinstance(other, ConfidenceScore):
            return self.value < other.value
        return NotImplemented

    def __gt__(self, other) -> bool:
        if isinstance(other, (int, float)):
            return self.value > other
        if isinstance(other, ConfidenceScore):
            return self.value > other.value
        return NotImplemented


# ============================================================================
# Language Code
# ============================================================================

class LanguageCode(str, Enum):
    """Supported language codes"""
    EN = "en"  # English
    KA = "ka"  # Georgian
    RU = "ru"  # Russian


# ============================================================================
# Pace
# ============================================================================

class Pace(str, Enum):
    """Trip pace options"""
    RELAXED = "relaxed"      # 2-3 activities per day
    MODERATE = "moderate"    # 3-4 activities per day
    INTENSIVE = "intensive"  # 5-6 activities per day


# ============================================================================
# Intent & Query Type
# ============================================================================

class Intent(str, Enum):
    """High-level user intent"""
    SEARCH = "SEARCH"           # Search for places
    PLAN = "PLAN"               # Plan itinerary
    ROUTE = "ROUTE"             # Get route between points
    INFORMATION = "INFORMATION" # General information


class QueryType(str, Enum):
    """Specific query type"""
    SIMPLE_SEARCH = "SIMPLE_SEARCH"           # "Find restaurants in Tbilisi"
    ITINERARY_PLANNING = "ITINERARY_PLANNING" # "Plan 3-day trip"
    ROUTE_QUERY = "ROUTE_QUERY"               # "Route from X to Y"
