"""
Itinerary schemas with tolerant validation and auto-correction.

Canonical itinerary shapes:
- Planning Agent  →  RawItineraryV1
- Geo Agent       →  EnrichedItineraryV1
"""

from pydantic import BaseModel, Field, model_validator
from typing import Optional, List
import logging

from schemas.common import Coordinates, ConfidenceScore

logger = logging.getLogger(__name__)


# ============================================================================
# Activity
# ============================================================================

class ActivityV1(BaseModel):
    """Single activity in itinerary"""
    name:        str             = Field(..., min_length=1)
    location:    str             = Field(..., min_length=1)
    category:    str             = Field(..., min_length=1)
    description: str             = Field("")
    coordinates: Optional[Coordinates] = Field(None)
    confidence:  ConfidenceScore = Field(default_factory=lambda: ConfidenceScore(value=1.0, source="default"))

    class Config:
        frozen = True


# ============================================================================
# Route
# ============================================================================

class RouteV1(BaseModel):
    """Route between two places"""
    from_place:   str   = Field(..., alias="from")
    to_place:     str   = Field(..., alias="to")
    distance_km:  float = Field(..., ge=0.0)
    duration_min: float = Field(..., ge=0.0)
    approximate:  bool  = Field(False)  # True when ORS failed → fallback used

    class Config:
        frozen = True
        populate_by_name = True


# ============================================================================
# Day
# ============================================================================

class DayV1(BaseModel):
    """
    Planning Agent output — no routes yet.
    Contains activities but no route information.
    """
    day:        int              = Field(..., ge=1)
    location:   str              = Field(..., min_length=1)
    activities: List[ActivityV1] = Field(..., min_items=1)

    class Config:
        frozen = True


class EnrichedDayV1(BaseModel):
    """
    Geo Agent output — routes present.
    Contains activities AND route information.

    Uses tolerant total validation for LLM-generated arithmetic.
    """
    day:               int               = Field(..., ge=1)
    location:          str               = Field(..., min_length=1)
    activities:        List[ActivityV1]  = Field(..., min_items=1)
    routes:            List[RouteV1]     = Field(default_factory=list)
    total_distance_km: float             = Field(0.0, ge=0.0)
    total_duration_min:float             = Field(0.0, ge=0.0)

    class Config:
        frozen = True

    @model_validator(mode="after")
    def _check_sums(self) -> "EnrichedDayV1":
        """
        Validate that totals match sum of routes.

        Auto-correct minor arithmetic differences and warn instead of failing
        the whole itinerary.
        """
        if not self.routes:
            return self

        # Calculate expected values
        expected_km  = round(sum(r.distance_km  for r in self.routes), 2)
        expected_min = round(sum(r.duration_min for r in self.routes), 2)

        # Check distance with RELAXED tolerance (1.0 km instead of 0.1 km)
        km_diff = abs(self.total_distance_km - expected_km)
        if km_diff > 1.0:
            logger.warning(
                f"Day {self.day}: Distance sum mismatch. "
                f"Reported: {self.total_distance_km} km, "
                f"Expected: {expected_km} km, "
                f"Diff: {km_diff:.2f} km. "
                f"Auto-correcting..."
            )

            # ✅ Auto-correct instead of raising
            # Note: object.__setattr__ works because validation happens AFTER init
            object.__setattr__(self, 'total_distance_km', expected_km)

        # Check duration with RELAXED tolerance (5 min instead of 0.1 min)
        min_diff = abs(self.total_duration_min - expected_min)
        if min_diff > 5.0:
            logger.warning(
                f"Day {self.day}: Duration sum mismatch. "
                f"Reported: {self.total_duration_min} min, "
                f"Expected: {expected_min} min, "
                f"Diff: {min_diff:.2f} min. "
                f"Auto-correcting..."
            )

            # ✅ Auto-correct
            object.__setattr__(self, 'total_duration_min', expected_min)

        return self


# ============================================================================
# Itinerary (Top-level)
# ============================================================================

class RawItineraryV1(BaseModel):
    """
    Planning Agent output. No route data.
    Contains only logical grouping of activities by day.
    """
    days:       List[DayV1] = Field(..., min_items=1)
    total_days: int         = Field(..., ge=1)
    pace:       str         = Field("moderate")
    summary:    str         = Field("")

    class Config:
        frozen = True

    @model_validator(mode="after")
    def _check_total(self) -> "RawItineraryV1":
        """Validate that total_days matches len(days)"""
        if self.total_days != len(self.days):
            raise ValueError(
                f"total_days={self.total_days} but len(days)={len(self.days)}"
            )
        return self


class EnrichedItineraryV1(BaseModel):
    """
    Geo Agent output. Has routes + distances.
    Contains complete trip information with routing.
    """
    days:       List[EnrichedDayV1] = Field(..., min_items=1)
    total_days: int                 = Field(..., ge=1)
    pace:       str                 = Field("moderate")
    summary:    str                 = Field("")

    class Config:
        frozen = True

    @model_validator(mode="after")
    def _check_total(self) -> "EnrichedItineraryV1":
        """Validate that total_days matches len(days)"""
        if self.total_days != len(self.days):
            raise ValueError(
                f"total_days={self.total_days} but len(days)={len(self.days)}"
            )
        return self

    @property
    def total_distance_km(self) -> float:
        """Total distance across all days"""
        return sum(day.total_distance_km for day in self.days)

    @property
    def total_duration_min(self) -> float:
        """Total duration across all days"""
        return sum(day.total_duration_min for day in self.days)

    @property
    def has_approximate_routes(self) -> bool:
        """Check if any routes are approximate (ORS API failed)"""
        for day in self.days:
            for route in day.routes:
                if route.approximate:
                    return True
        return False
