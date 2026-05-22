"""LLM invocation helpers with retry and budget tracking."""

import time
import asyncio
import logging
from typing import Any, Tuple, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from config.settings import get_settings

from llm.policies import LLMPolicy, get_policy

logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================

class BudgetExceededError(Exception):
    """Budget exceeded during LLM call"""

    def __init__(
        self,
        policy_name: str,
        reason: str,
        elapsed_s: float = 0.0,
        tokens_used: int = 0
    ):
        self.policy_name = policy_name
        self.reason = reason
        self.elapsed_s = elapsed_s
        self.tokens_used = tokens_used
        super().__init__(
            f"Budget exceeded for policy '{policy_name}': "
            f"reason={reason}, elapsed={elapsed_s:.2f}s, tokens={tokens_used}"
        )


# ============================================================================
# LLM Building
# ============================================================================

def build_llm(policy_name: str) -> ChatOpenAI:
    """
    Build LLM with policy settings.

    Args:
        policy_name: Policy name (e.g., "search_classifier")

    Returns:
        Configured ChatOpenAI instance
    """
    policy = get_policy(policy_name)
    api_key = get_settings().openai_api_key

    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")

    return ChatOpenAI(
        model=policy.model,
        temperature=policy.temperature,
        max_tokens=policy.max_tokens,
        api_key=api_key,
        timeout=policy.timeout_seconds,
        max_retries=0,
    )


def build_llm_structured(policy_name: str, output_schema: Any) -> Any:
    """Build LLM with structured output"""
    return build_llm(policy_name).with_structured_output(output_schema, method="function_calling")


# ============================================================================
# Retry Logic
# ============================================================================

def _is_retryable(exc: Exception) -> bool:
    """
    Check if exception is retryable.

    Retryable:
    - Timeout errors
    - Rate limit errors (429)
    - Server errors (503)
    """
    msg = str(exc).lower()
    return (
        isinstance(exc, asyncio.TimeoutError)
        or "timeout" in msg
        or "rate" in msg
        or "429" in msg
        or "503" in msg
        or "service unavailable" in msg
    )


async def ainvoke_with_retry(
    llm: ChatOpenAI,
    messages: list[BaseMessage],
    policy: LLMPolicy
) -> Any:
    """
    Invoke LLM with exponential backoff retry.

    Args:
        llm: ChatOpenAI instance
        messages: Messages to send
        policy: LLM policy

    Returns:
        LLM response

    Raises:
        Exception: If all retries failed
    """
    last_exc: Exception = RuntimeError("No attempts made")

    for attempt in range(policy.max_retries + 1):
        try:
            return await llm.ainvoke(messages)

        except Exception as e:
            last_exc = e

            # Don't retry if not retryable or last attempt
            if not _is_retryable(e) or attempt == policy.max_retries:
                raise

            # Exponential backoff
            delay = policy.retry_base_delay_s * (2 ** attempt)
            logger.warning(
                f"LLM call failed (attempt {attempt + 1}/{policy.max_retries + 1}), "
                f"retrying in {delay:.1f}s: {e}"
            )

            await asyncio.sleep(delay)

    raise last_exc


# ============================================================================
# Budget-Aware Invocation
# ============================================================================

async def ainvoke_with_budget(
    llm: ChatOpenAI,
    messages: list[BaseMessage],
    policy: LLMPolicy
) -> Tuple[Any, dict]:
    """
    Invoke LLM with budget control.

    Args:
        llm: ChatOpenAI instance
        messages: Messages to send
        policy: LLM policy

    Returns:
        (response, usage_dict)
        usage_dict = {
            "prompt_tokens": int,
            "completion_tokens": int,
            "total_tokens": int,
            "elapsed_ms": float,
            "model": str
        }

    Raises:
        BudgetExceededError: If timeout exceeded
    """
    effective_timeout = policy.budget_max_seconds or policy.timeout_seconds
    t0 = time.monotonic()

    # Execute with timeout
    try:
        response = await asyncio.wait_for(
            ainvoke_with_retry(llm, messages, policy),
            timeout=effective_timeout
        )
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - t0
        raise BudgetExceededError(
            policy_name=policy.name,
            reason="time",
            elapsed_s=elapsed
        )

    # Extract usage
    elapsed_ms = (time.monotonic() - t0) * 1000
    usage = getattr(response, "response_metadata", {}).get("token_usage", {})

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    # Completion-token budget is checked after the call for monitoring.
    if policy.budget_max_tokens is not None:
        if completion_tokens > policy.budget_max_tokens:
            logger.error(
                f"Budget exceeded (post-factum): policy={policy.name}, "
                f"completion_tokens={completion_tokens}, "
                f"budget={policy.budget_max_tokens}. "
                f"NOTE: OpenAI should have cut at {policy.max_tokens} (max_tokens), "
                f"this indicates a bug!"
            )
        elif completion_tokens > policy.budget_max_tokens * 0.9:
            logger.warning(
                f"Approaching budget limit: policy={policy.name}, "
                f"completion_tokens={completion_tokens}, "
                f"budget={policy.budget_max_tokens}"
            )

    # Return usage for state update
    usage_dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "elapsed_ms": elapsed_ms,
        "model": policy.model
    }

    return response, usage_dict


# ============================================================================
# Convenience Wrapper
# ============================================================================

async def invoke_with_policy(
    policy_name: str,
    messages: list[BaseMessage],
    output_schema: Optional[Any] = None
) -> Tuple[Any, dict]:
    """
    Convenience wrapper: build LLM + invoke with budget.

    Args:
        policy_name: Policy name
        messages: Messages to send
        output_schema: Optional schema for structured output

    Returns:
        (response, usage_dict)

    Example:
        response, usage = await invoke_with_policy(
            "search_classifier",
            [HumanMessage(content="Find restaurants")]
        )

        # Update state budget
        new_budget = state.budget_state.add_llm_call(
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            model=usage["model"]
        )
        new_state = state.with_budget_update(new_budget)
    """
    policy = get_policy(policy_name)
    llm = build_llm(policy_name)

    if output_schema:
        # Call the base LLM to collect usage, then parse structured output.
        response, usage_dict = await ainvoke_with_budget(llm, messages, policy)
        structured_llm = llm.with_structured_output(output_schema, method="function_calling")
        from langchain_core.messages import AIMessage
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # Parse tool_call arguments directly.
            import json
            tool_args = response.tool_calls[0].get('args', {})
            parsed = output_schema(**tool_args)
        else:
            # Fallback: call the structured LLM separately without usage tracking.
            parsed = await structured_llm.ainvoke(messages)
        return parsed, usage_dict
    else:
        return await ainvoke_with_budget(llm, messages, policy)
