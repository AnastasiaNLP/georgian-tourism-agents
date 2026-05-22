"""
Utilities for managing main graph messages.

Rules:
1. Agents do not write internal messages to main state.
   Main state["messages"] contains only orchestrator conversation.
   
2. If messages grows too large, old context is compacted.
"""

import logging
from typing import Any
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger(__name__)

# Maximum messages in orchestrator context before trimming.
MAX_MESSAGES = 10


def should_trim(messages: list) -> bool:
    """Return whether message trimming is needed."""
    return len(messages) > MAX_MESSAGES


def trim_messages(messages: list) -> list:
    """
    Compact old messages when the context is too large.
    
    Strategy: keep the first user query and the last 4 messages, then replace
    the middle with one summary placeholder.
    
    This uses a simple placeholder instead of LLM summarization.
    """
    if not should_trim(messages):
        return messages
    
    logger.info(f"Trimming messages: {len(messages)} → keeping first + last 4")
    
    # Original user query.
    first = messages[0]
    
    # Most recent context.
    recent = messages[-4:]
    
    # Summary placeholder for the trimmed middle section.
    trimmed_count = len(messages) - 5
    summary = AIMessage(
        content=f"[... {trimmed_count} messages trimmed for context efficiency ...]"
    )
    
    return [first, summary] + recent


def add_orchestrator_message(messages: list, content: str) -> list:
    """
    Add an orchestrator message without mutating the original list.
    """
    new_msg = AIMessage(content=content)
    new_messages = list(messages) + [new_msg]
    
    # Trim if needed.
    if should_trim(new_messages):
        new_messages = trim_messages(new_messages)
    
    return new_messages
