"""Validation schemas used by ValidationAgent."""

from pydantic import BaseModel, Field
from typing import List, Optional

from schemas.common import ConfidenceScore


# ============================================================================
# Issue & Suggestion
# ============================================================================

class IssueV1(BaseModel):
    """
    One concrete problem found during validation.

    `day` scopes it to specific day; None = whole trip.
    """
    day:      Optional[int] = Field(None, description="Day number (None = whole trip)")
    field:    str           = Field(..., min_length=1, description="Which field has issue")
    message:  str           = Field(..., min_length=1, description="What's wrong")
    severity: str           = Field("error", description="error | warning")

    class Config:
        frozen = True


class SuggestionV1(BaseModel):
    """Suggested fix for an issue"""
    day:     Optional[int] = Field(None, description="Day number (None = whole trip)")
    action:  str           = Field(..., min_length=1, description="What to do")
    message: str           = Field(..., min_length=1, description="Why do it")

    class Config:
        frozen = True


# ============================================================================
# Feasibility Report
# ============================================================================

class FeasibilityReportV1(BaseModel):
    """
    Validation Agent output.

    Result of a route feasibility check.
    """
    feasible:    bool                     = Field(..., description="Is itinerary feasible?")
    score:       ConfidenceScore          = Field(default_factory=lambda: ConfidenceScore(value=1.0, source="validation"))
    issues:      List[IssueV1]            = Field(default_factory=list, description="Blocking issues")
    warnings:    List[IssueV1]            = Field(default_factory=list, description="Non-blocking warnings")
    suggestions: List[SuggestionV1]       = Field(default_factory=list, description="How to improve")
    summary:     str                      = Field("", description="Overall assessment")

    class Config:
        frozen = True

    @property
    def has_critical_issues(self) -> bool:
        """Any critical (error) issues?"""
        return any(issue.severity == "error" for issue in self.issues)

    @property
    def issue_count(self) -> int:
        """Total number of issues"""
        return len(self.issues)

    @property
    def warning_count(self) -> int:
        """Total number of warnings"""
        return len(self.warnings)


# ============================================================================
# Validation result stored in state.
# ============================================================================

class ValidationResult(BaseModel):
    """
    Validation result returned by ValidationAgent.

    Includes recommended_action for routing.
    """
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    severity: str = Field("info", description="info | warning | critical")

    # Recommended action for routing.
    recommended_action: str = Field("proceed", description="proceed | retry | degrade | abort")

    # Optional: full report
    report: Optional[FeasibilityReportV1] = None

    class Config:
        frozen = True
