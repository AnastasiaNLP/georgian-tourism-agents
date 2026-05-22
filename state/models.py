"""Pydantic models for agent and orchestrator validation."""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class ExecutionMode(str, Enum):
    NORMAL = "normal"
    DEGRADED = "degraded"
    EMERGENCY = "emergency"


class BudgetState(BaseModel):
    """
    Budget tracking for a single request.
    """
    llm_calls: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0

    # Approximate provider prices.
    _COSTS = {
        "claude-sonnet-4-20250514": {"prompt": 0.003,   "completion": 0.015},
        "gpt-4o-mini":              {"prompt": 0.00015, "completion": 0.0006},
        "gpt-4o":                   {"prompt": 0.0025,  "completion": 0.01},
    }

    def add_llm_call(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "gpt-4o-mini"
    ) -> "BudgetState":
        """Return a new budget with one LLM call added."""
        costs = self._COSTS.get(model, self._COSTS["gpt-4o-mini"])
        cost_usd = (
            (prompt_tokens / 1000) * costs["prompt"] +
            (completion_tokens / 1000) * costs["completion"]
        )
        return BudgetState(
            llm_calls=self.llm_calls + 1,
            tool_calls=self.tool_calls,
            prompt_tokens=self.prompt_tokens + prompt_tokens,
            completion_tokens=self.completion_tokens + completion_tokens,
            estimated_cost_usd=round(self.estimated_cost_usd + cost_usd, 6),
        )

    def add_tool_call(self) -> "BudgetState":
        return BudgetState(
            llm_calls=self.llm_calls,
            tool_calls=self.tool_calls + 1,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            estimated_cost_usd=self.estimated_cost_usd,
        )

    def is_over_budget(
        self,
        max_cost_usd: float = 0.50,
        max_llm_calls: int = 15,
    ) -> tuple[bool, str]:
        if self.estimated_cost_usd >= max_cost_usd:
            return True, f"cost ${self.estimated_cost_usd:.4f} >= ${max_cost_usd}"
        if self.llm_calls >= max_llm_calls:
            return True, f"llm_calls {self.llm_calls} >= {max_llm_calls}"
        return False, ""

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict) -> "BudgetState":
        return cls(**d) if d else cls()


# ── Orchestrator structured output models ────────────────────────────────────

class OrchestratorStep(BaseModel):
    """One orchestrator plan step."""
    agent: str = Field(description="Agent name: search_agent | planning_agent | geo_agent | validation_agent | response_agent")
    params: dict = Field(default_factory=dict, description="Parameters for the agent, e.g. {region: 'Adjara'}")
    reason: str = Field(default="", description="Why this agent is needed")


class OrchestratorPlan(BaseModel):
    """Structured output for the orchestrator classifier."""
    steps: List[OrchestratorStep] = Field(description="Ordered list of agents to call")
    reasoning: str = Field(description="Why this plan was chosen")
    estimated_agents: int = Field(description="Total number of agent calls expected")


class OrchestratorDecision(BaseModel):
    """Structured output for optional orchestrator review decisions."""
    next_action: str = Field(
        description="One of: call_agent | retry_agent | respond | done"
    )
    agent: Optional[str] = Field(
        default=None,
        description="Agent to call next (if next_action=call_agent|retry_agent)"
    )
    params: Optional[dict] = Field(
        default=None,
        description="Parameters for next agent"
    )
    feedback: Optional[str] = Field(
        default=None,
        description="Feedback for retry (if next_action=retry_agent)"
    )
    reasoning: str = Field(description="Why this decision was made")
