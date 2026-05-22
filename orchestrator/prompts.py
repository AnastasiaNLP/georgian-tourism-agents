# orchestrator/prompts.py
"""System prompt loader for YAML prompt files."""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Loaded prompt cache: file stem -> prompt text.
_cache: dict[str, str] = {}


def load_prompt(name: str) -> str:
    """
    Load a prompt by name.

    Looks for config/prompts/{name}.yaml and falls back to a built-in prompt.

    Args:
        name: Prompt name without extension, for example "orchestrator".

    Returns:
        Prompt text.
    """
    if name in _cache:
        return _cache[name]

    # Resolve relative to repository root.
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(base_dir, "config", "prompts", f"{name}.yaml")

    if not os.path.exists(yaml_path):
        logger.warning(f"Prompt not found: {yaml_path}; using fallback")
        return _fallback_prompt(name)

    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # YAML can contain either a raw string or a dict with a "system" key.
        if isinstance(data, str):
            text = data
        elif isinstance(data, dict):
            text = data.get("system", data.get("prompt", str(data)))
        else:
            text = str(data)

        _cache[name] = text
        logger.info(f"Prompt loaded: {yaml_path} ({len(text)} chars)")
        return text

    except Exception as e:
        logger.error(f"Prompt load failed {yaml_path}: {e}")
        return _fallback_prompt(name)


def _fallback_prompt(name: str) -> str:
    """Built-in fallback prompt used when YAML is missing."""
    if name == "orchestrator":
        return (
            "You are an orchestrator for a Georgian tourism planning system. "
            "Analyze the user query and create a plan using available agents: "
            "search_agent, planning_agent, geo_agent, validation_agent, response_agent. "
            "Always respond with valid JSON matching the required schema."
        )
    return f"You are a helpful assistant for the {name} task."


def clear_cache() -> None:
    """Clear prompt cache for tests."""
    _cache.clear()
