"""
Execution guard for workflow safety limits.

If a limit is exceeded, the graph can skip orchestration and move to the
response agent so the request still terminates.
"""

import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GuardResult:
    """Execution guard check result."""
    allowed: bool
    reason: str
    force_response: bool


class ExecutionGuard:
    """
    Checks request-level safety limits before orchestration.
    """

    def __init__(
        self,
        max_orchestrator_calls: int = 8,
        max_agent_retries: int = 2,
        max_wall_time_seconds: float = 300.0,
        max_cost_usd: float = 0.50,
    ):
        self.max_orchestrator_calls = max_orchestrator_calls
        self.max_agent_retries = max_agent_retries
        self.max_wall_time_seconds = max_wall_time_seconds
        self.max_cost_usd = max_cost_usd

    def check_before_plan(self, state: dict) -> GuardResult:
        """
        Check whether the initial orchestrator call is allowed.
        """
        return self._check(state, context="plan")

    def check_before_review(self, state: dict) -> GuardResult:
        """
        Check whether an optional review call is allowed.
        """
        return self._check(state, context="review")

    def _check(self, state: dict, context: str) -> GuardResult:
        """Run all guard checks."""

        # Orchestrator call count.
        trace = state.get("router_trace") or []
        orchestrator_calls = len([
            t for t in trace
            if t.get("from", "").startswith("orchestrator")
        ])
        if orchestrator_calls >= self.max_orchestrator_calls:
            return GuardResult(
                allowed=False,
                reason=f"orchestrator calls {orchestrator_calls} >= {self.max_orchestrator_calls}",
                force_response=True,
            )

        # Per-agent retry count.
        history = state.get("agent_history") or []
        if history:
            from collections import Counter
            counts = Counter(history)
            most_called = counts.most_common(1)[0]
            if most_called[1] > self.max_agent_retries:
                return GuardResult(
                    allowed=False,
                    reason=f"agent {most_called[0]} retried {most_called[1]} times",
                    force_response=True,
                )

        # Wall-clock request duration.
        start_time = state.get("execution_start_time")
        if start_time:
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time)
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            if elapsed >= self.max_wall_time_seconds:
                return GuardResult(
                    allowed=False,
                    reason=f"wall time {elapsed:.1f}s >= {self.max_wall_time_seconds}s",
                    force_response=True,
                )

        # Estimated provider cost.
        budget = state.get("budget_state") or {}
        cost = budget.get("estimated_cost_usd", 0.0)
        if cost >= self.max_cost_usd:
            return GuardResult(
                allowed=False,
                reason=f"cost ${cost:.4f} >= ${self.max_cost_usd}",
                force_response=True,
            )

        # Emergency mode bypasses orchestration.
        mode = state.get("execution_mode", "normal")
        if mode == "emergency":
            return GuardResult(
                allowed=False,
                reason="execution_mode=emergency, skipping orchestrator",
                force_response=True,
            )

        return GuardResult(allowed=True, reason="ok", force_response=False)
