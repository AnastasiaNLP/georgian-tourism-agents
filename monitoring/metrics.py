"""Prometheus metrics for Georgian Tourism Agent."""

import time
import logging
from functools import wraps
from typing import Callable

logger = logging.getLogger(__name__)

# Fall back to no-op metrics when prometheus_client is not installed.
try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

    REGISTRY = CollectorRegistry()

    AGENT_DURATION = Histogram(
        "agent_duration_seconds",
        "Agent execution duration in seconds",
        ["agent_name"],
        registry=REGISTRY,
        buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
    )

    AGENT_CALLS = Counter(
        "agent_calls_total",
        "Total agent calls",
        ["agent_name", "status"],  # status: success | error | timeout
        registry=REGISTRY,
    )

    ORCHESTRATOR_DECISIONS = Counter(
        "orchestrator_decisions_total",
        "Orchestrator review decisions",
        ["decision"],  # call_agent | retry_agent | respond | done
        registry=REGISTRY,
    )

    ACTIVE_REQUESTS = Gauge(
        "active_requests",
        "Number of active planning requests",
        registry=REGISTRY,
    )

    LLM_TOKENS = Counter(
        "llm_tokens_total",
        "LLM tokens used",
        ["agent_name", "token_type"],  # token_type: prompt | completion
        registry=REGISTRY,
    )

    _PROMETHEUS_AVAILABLE = True
    logger.info("Prometheus metrics initialized")

except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed; metrics disabled")

    # No-op metric objects.
    class _Noop:
        def labels(self, **kwargs): return self
        def observe(self, v): pass
        def inc(self, v=1): pass
        def dec(self, v=1): pass
        def set(self, v): pass

    AGENT_DURATION = _Noop()
    AGENT_CALLS = _Noop()
    ORCHESTRATOR_DECISIONS = _Noop()
    ACTIVE_REQUESTS = _Noop()
    LLM_TOKENS = _Noop()
    REGISTRY = None


# ── Decorator ─────────────────────────────────────────────────────────────────

def track_agent(agent_name: str) -> Callable:
    """
    Decorator for agent nodes that records metrics automatically.

    Records duration and success/error/timeout counters.

    Usage:
        @track_agent("search_agent")
        async def search_agent_node(state): ...

        # Existing functions can be wrapped directly.
        tracked_node = track_agent("search_agent")(existing_node)
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        async def wrapper(state: dict) -> dict:
            start = time.monotonic()
            status = "success"
            try:
                result = await fn(state)
                # Determine status from errors in the returned partial state.
                if result.get("errors"):
                    status = "error"
                return result
            except Exception as e:
                status = "timeout" if "timeout" in str(e).lower() else "error"
                raise
            finally:
                duration = time.monotonic() - start
                AGENT_DURATION.labels(agent_name=agent_name).observe(duration)
                AGENT_CALLS.labels(agent_name=agent_name, status=status).inc()
                logger.debug(
                    f"[metrics] {agent_name}: {duration:.2f}s, status={status}"
                )
        return wrapper
    return decorator


def record_orchestrator_decision(decision: str) -> None:
    """Record an orchestrator routing decision."""
    ORCHESTRATOR_DECISIONS.labels(decision=decision).inc()


def record_llm_tokens(agent_name: str, prompt: int, completion: int) -> None:
    """Record token usage."""
    LLM_TOKENS.labels(agent_name=agent_name, token_type="prompt").inc(prompt)
    LLM_TOKENS.labels(agent_name=agent_name, token_type="completion").inc(completion)


def is_available() -> bool:
    """Return whether Prometheus metrics are available."""
    return _PROMETHEUS_AVAILABLE
