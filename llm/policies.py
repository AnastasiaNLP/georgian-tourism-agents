"""LLM policies for agents and execution modes."""

from dataclasses import dataclass
from typing import Optional, Dict

# execution_mode is one of: normal, degraded, emergency.


# ============================================================================
# LLM Policy
# ============================================================================

@dataclass(frozen=True)
class LLMPolicy:
    """
    Configuration for LLM calls.
    """
    name:                str
    model:               str   = "gpt-4o-mini"
    temperature:         float = 0.0
    max_tokens:          int   = 2000
    timeout_seconds:     int   = 30
    max_retries:         int   = 3
    retry_base_delay_s:  float = 1.0

    # Budget limits (optional)
    budget_max_tokens:   Optional[int]   = None
    budget_max_seconds:  Optional[float] = None


# ============================================================================
# Default Policies (NORMAL mode)
# ============================================================================

SEARCH_POLICY_V1 = LLMPolicy(
    name="search_classifier",
    model="gpt-4o-mini",
    temperature=0.0,
    max_tokens=500,
    timeout_seconds=15,
    max_retries=3,
    budget_max_tokens=500,
    budget_max_seconds=12.0,
)

PLANNING_POLICY_V1 = LLMPolicy(
    name="planning",
    model="gpt-4o-mini",
    temperature=0.3,
    max_tokens=2000,
    timeout_seconds=30,
    max_retries=2,
    budget_max_tokens=2000,
    budget_max_seconds=25.0,
)

GEO_FALLBACK_POLICY_V1 = LLMPolicy(
    name="geo_fallback",
    model="gpt-4o-mini",
    temperature=0.0,
    max_tokens=200,
    timeout_seconds=10,
    max_retries=1,
    budget_max_tokens=200,
    budget_max_seconds=8.0,
)

VALIDATION_POLICY_V1 = LLMPolicy(
    name="validation",
    model="gpt-4o-mini",
    temperature=0.0,
    max_tokens=1000,
    timeout_seconds=20,
    max_retries=2,
    budget_max_tokens=1000,
    budget_max_seconds=18.0,
)

RESPONSE_POLICY_V1 = LLMPolicy(
    name="response_generator",
    model="gpt-4o-mini",
    temperature=0.4,
    max_tokens=3000,
    timeout_seconds=30,
    max_retries=2,
    budget_max_tokens=3000,
    budget_max_seconds=28.0,
)


# ============================================================================
# Policy Registry
# ============================================================================

_REGISTRY: Dict[str, LLMPolicy] = {
    p.name: p for p in [
        SEARCH_POLICY_V1,
        PLANNING_POLICY_V1,
        GEO_FALLBACK_POLICY_V1,
        VALIDATION_POLICY_V1,
        RESPONSE_POLICY_V1,
    ]
}


def get_policy(name: str) -> LLMPolicy:
    """
    Get policy by name.

    Args:
        name: Policy name

    Returns:
        LLMPolicy

    Raises:
        KeyError: If policy not found
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"LLM policy '{name}' not found. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


# ============================================================================
# Execution Mode Adaptors
# ============================================================================



# ============================================================================
# Orchestrator Policy
# ============================================================================

ORCHESTRATOR_POLICY = LLMPolicy(
    name="orchestrator",
    model="claude-haiku-4-5-20251001",
    temperature=0.0,
    max_tokens=1000,
    timeout_seconds=30,
    max_retries=2,
    budget_max_tokens=1000,
    budget_max_seconds=25.0,
)

_REGISTRY["orchestrator"] = ORCHESTRATOR_POLICY
