# schemas/request.py
"""Canonical input contract for API requests."""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List

from schemas.common import LanguageCode, Pace


# ============================================================================
# Trip Parameters
# ============================================================================

class TripParametersV1(BaseModel):
    """
    Trip planning parameters.

    Structured trip planning parameters.
    """
    days:           int           = Field(..., ge=1, le=14, description="Number of days (1-14)")
    start_location: str           = Field(..., min_length=1, description="Starting location")
    end_location:   Optional[str] = Field(None, description="Ending location (optional)")
    pace:           str           = Field("moderate", description="Trip pace: relaxed, moderate, intensive")
    interests:      List[str]     = Field(default_factory=list, description="List of interests")

    class Config:
        frozen = True

    @field_validator("pace")
    @classmethod
    def validate_pace(cls, v: str) -> str:
        """Normalize pace synonyms to supported values."""
        valid_paces = {"relaxed", "moderate", "intensive"}
        if v in valid_paces:
            return v
        # Synonyms that may come from an LLM or UI.
        mapping = {
            "self-drive": "moderate", "easy": "relaxed", "slow": "relaxed",
            "fast": "intensive", "quick": "intensive", "busy": "intensive",
            "normal": "moderate", "medium": "moderate", "standard": "moderate"
        }
        return mapping.get(v.lower(), "moderate")
        return v

    @field_validator("interests", mode="before")
    @classmethod
    def normalize_interests(cls, v) -> list:
        """Convert comma-separated strings to lists."""
        if isinstance(v, str):
            return [i.strip() for i in v.split(",") if i.strip()]
        return [i.strip().lower() for i in v if i.strip()]


# ============================================================================
# User Request
# ============================================================================

class UserRequestV1(BaseModel):
    """
    Single entry-point schema.

    Shared request schema for API, chat, and CLI entry points.
    """
    query:           str                        = Field(..., min_length=1, description="User query")
    language:        str                        = Field("en", description="Response language")
    trip_parameters: Optional[TripParametersV1] = Field(None, description="Trip parameters (optional)")

    # Populated by the session layer, not by the user.
    request_id:     Optional[str] = Field(None, description="Request ID (auto-generated)")
    session_id:     Optional[str] = Field(None, description="Session ID (auto-generated)")
    correlation_id: Optional[str] = Field(None, description="Correlation ID (auto-generated)")

    class Config:
        frozen = True

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Validate language code"""
        valid_languages = {"en", "ka", "ru"}
        if v not in valid_languages:
            raise ValueError(f"language must be one of {valid_languages}")
        return v

    def to_state_input(self) -> dict:
        """
        Convert to state initialization kwargs.

        Returns:
            Dict for creating TravelPlanningState.
        """
        return {
            "user_query":      self.query,
            "user_language":   self.language,
            "trip_parameters": self.trip_parameters.model_dump() if self.trip_parameters else None,
            "request_id":      self.request_id or "",
            "correlation_id":  self.correlation_id or "",
        }
