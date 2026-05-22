"""
safe_node() wraps graph nodes with timeout and error cascade handling.

All agents registered through AgentRegistry automatically get this protection.
"""

import asyncio
import logging
import traceback
from typing import Callable, Optional
from functools import wraps

logger = logging.getLogger(__name__)

# Timeout per agent in seconds.
AGENT_TIMEOUTS = {
    "search_agent":     60,
    "planning_agent":   90,
    "geo_agent":        60,
    "validation_agent": 45,
    "response_agent":   90,
}


def safe_node(fn: Callable, agent_name: str) -> Callable:
    """
    Wrap a node with timeout and error cascade handling.

    On timeout or exception, return a partial state update instead of raising.
    """
    timeout = AGENT_TIMEOUTS.get(agent_name, 40)

    @wraps(fn)
    async def wrapper(state: dict) -> dict:
        try:
            return await asyncio.wait_for(fn(state), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"[{agent_name}] timeout {timeout}s")
            return _escalate(state, agent_name, f"timeout {timeout}s",
                             error_type="TimeoutError")
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[{agent_name}] exception: {e}\n{tb}")
            return _escalate(state, agent_name, str(e)[:200],
                             error_type=type(e).__name__)

    return wrapper


TRANSIENT_ERRORS = {"TimeoutError", "ConnectionError", "RateLimitError", "asyncio.TimeoutError"}


def _escalate(state: dict, agent_name: str, reason: str, error_type: str = "unknown") -> dict:
    """
    Error cascade: normal → degraded → emergency.
    Transient errors in normal mode keep the workflow in normal mode.
    Permanent or repeated errors degrade execution mode.
    """
    current = state.get("execution_mode", "normal")

    is_transient = any(t in str(error_type) for t in TRANSIENT_ERRORS)

    if is_transient and current == "normal":
        # Log transient errors without degrading the first time.
        logger.warning(f"[{agent_name}] transient error, staying normal: {reason}")
        new_mode = "normal"
    else:
        new_mode = "degraded" if current == "normal" else "emergency"
        logger.warning(f"[{agent_name}] cascade: {current} → {new_mode} | {reason}")

    return {
        "errors":          [f"{agent_name}: {reason}"],
        "execution_mode":  new_mode,
        "agent_history":   [agent_name],
        "agent_scratchpad": {
            "agent":   agent_name,
            "summary": f"FAILED: {reason[:100]}",
            "error":   True,
        },
    }
