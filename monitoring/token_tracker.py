"""Token tracking through LangChain callbacks."""
import logging
from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)

# Cost per 1,000 tokens.
_COSTS = {
    "gpt-4o-mini":               {"prompt": 0.00015,  "completion": 0.0006},
    "claude-haiku-4-5-20251001": {"prompt": 0.001,    "completion": 0.005},
}


class TokenTracker(BaseCallbackHandler):
    """Track tokens and tool calls for one agent invocation."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.llm_calls = 0
        self.tool_calls = 0

    def on_llm_end(self, response, **kwargs):
        """Called after each LLM call."""
        self.llm_calls += 1
        usage = getattr(response, "llm_output", {}) or {}
        if isinstance(usage, dict):
            token_usage = usage.get("token_usage", {}) or {}
            self.prompt_tokens    += token_usage.get("prompt_tokens", 0)
            self.completion_tokens += token_usage.get("completion_tokens", 0)

    def on_tool_start(self, serialized, input_str, **kwargs):
        """Called on each tool call."""
        self.tool_calls += 1

    def get_cost_usd(self) -> float:
        """Estimate cost based on the model."""
        c = _COSTS.get(self.model, _COSTS["gpt-4o-mini"])
        return (
            (self.prompt_tokens    / 1000) * c["prompt"] +
            (self.completion_tokens / 1000) * c["completion"]
        )

    def to_dict(self) -> dict:
        return {
            "llm_calls":         self.llm_calls,
            "tool_calls":        self.tool_calls,
            "prompt_tokens":     self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


def merge_budget(current: dict, tracker: TokenTracker) -> dict:
    """
    Add tracker usage to the current budget_state.
    """
    return {
        "llm_calls":          current.get("llm_calls", 0)          + tracker.llm_calls,
        "tool_calls":         current.get("tool_calls", 0)         + tracker.tool_calls,
        "prompt_tokens":      current.get("prompt_tokens", 0)      + tracker.prompt_tokens,
        "completion_tokens":  current.get("completion_tokens", 0)  + tracker.completion_tokens,
        "estimated_cost_usd": current.get("estimated_cost_usd", 0) + tracker.get_cost_usd(),
    }
