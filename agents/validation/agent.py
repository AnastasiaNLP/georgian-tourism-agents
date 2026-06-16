# agents/validation/agent.py
"""
ValidationAgent performs programmatic itinerary checks.
The LLM is used only for warnings when explicitly enabled.

Node wrapper: validation_agent_node(state: dict) → dict
"""
from __future__ import annotations
import logging
import re

from agents.base import extract_pace, make_scratchpad

logger = logging.getLogger(__name__)


async def validation_agent_node(state: dict) -> dict:
    """Run programmatic validation, with optional LLM review for warnings."""
    from monitoring.token_tracker import TokenTracker, merge_budget

    request_id = state.get("request_id", "unknown")
    itinerary = state.get("enriched_itinerary") or state.get("raw_itinerary")

    if not itinerary:
        return {
            "validation_result": {
                "is_valid": False, "score": 0.0,
                "errors": ["No itinerary to validate"],
                "warnings": [], "recommended_action": "abort",
            },
            "agent_history": ["validation_agent"],
            "agent_scratchpad": make_scratchpad("validation_agent", "SKIPPED: no itinerary"),
        }

    pace = extract_pace(state)
    distances = state.get("distance_matrix") or {}
    history = state.get("agent_history") or []
    planning_count = history.count("planning_agent")

    # First pass: deterministic validation.
    result = _auto_validate(itinerary, pace, distances)

    if not result["errors"] and not result["warnings"]:
        logger.info(f"[{request_id}] validation_agent: auto PASSED (0 LLM calls)")
        return {
            "validation_result": result,
            "agent_history": ["validation_agent"],
            "agent_scratchpad": make_scratchpad(
                "validation_agent",
                f"PASSED (programmatic): score={result['score']}",
                is_valid=True, action="proceed"
            ),
        }

    # Hard errors do not need LLM review.
    if result["errors"]:
        is_degraded = planning_count >= 2
        if is_degraded:
            result["recommended_action"] = "proceed"
        logger.info(
            f"[{request_id}] validation_agent: FAILED programmatic, "
            f"{len(result['errors'])} errors"
            f"{' (retry exhausted → degraded)' if is_degraded else ' (→ retry)'}"
        )
        return_dict = {
            "validation_result": result,
            "agent_history": ["validation_agent"],
            "agent_scratchpad": make_scratchpad(
                "validation_agent",
                f"FAILED: {result['errors'][:2]}",
                is_valid=False, action=result["recommended_action"]
            ),
        }
        if is_degraded:
            return_dict["execution_mode"] = "degraded"
            return_dict["errors"] = [f"validation: {e}" for e in result["errors"][:3]]
        elif result["recommended_action"] == "retry":
            return_dict["raw_itinerary"] = {}
            return_dict["enriched_itinerary"] = {}
        return return_dict

    # Warnings may use one optional LLM call for a final verdict.
    tracker = TokenTracker(model="gpt-4o-mini")
    try:
        flags = state.get("feature_flags") or {}
        if not flags.get("ENABLE_VALIDATION_LLM", False):
            logger.info(f"[{request_id}] validation_agent: warnings only, validation LLM disabled")
            return {
                "validation_result": result,
                "agent_history": ["validation_agent"],
                "budget_state": merge_budget(state.get("budget_state") or {}, tracker),
                "agent_scratchpad": make_scratchpad(
                    "validation_agent",
                    f"PASSED with warnings (programmatic): score={result['score']}",
                    is_valid=result["is_valid"], action=result["recommended_action"]
                ),
            }

        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        from config.settings import get_settings
        import json

        api_key = get_settings().openai_api_key
        if not api_key:
            logger.info(f"[{request_id}] validation_agent: warnings only, no OPENAI_API_KEY; skipping LLM")
            return {
                "validation_result": result,
                "agent_history": ["validation_agent"],
                "budget_state": merge_budget(state.get("budget_state") or {}, tracker),
                "agent_scratchpad": make_scratchpad(
                    "validation_agent",
                    f"PASSED with warnings (programmatic): score={result['score']}",
                    is_valid=result["is_valid"], action=result["recommended_action"]
                ),
            }

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=400,
                         api_key=api_key, timeout=60)
        prompt = (
            f"Review these travel itinerary warnings and decide if they are acceptable:\n"
            f"WARNINGS: {result['warnings']}\n"
            f"PACE: {pace}\n"
            f"Respond ONLY with JSON: "
            '{"acceptable": true/false, "reason": "brief"}'
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)],
                                      config={"callbacks": [tracker]})
        parsed = json.loads(response.content.strip())
        if not parsed.get("acceptable", True):
            result["recommended_action"] = "retry" if planning_count < 2 else "proceed"
            result["is_valid"] = False

    except Exception as exc:
        logger.warning(f"validation LLM review failed: {exc}")

    from monitoring.token_tracker import merge_budget
    return {
        "validation_result": result,
        "agent_history": ["validation_agent"],
        "budget_state": merge_budget(state.get("budget_state") or {}, tracker),
        "agent_scratchpad": make_scratchpad(
            "validation_agent",
            f"{'PASSED' if result['is_valid'] else 'WARNINGS'}: "
            f"score={result['score']}, action={result['recommended_action']}",
            is_valid=result["is_valid"], action=result["recommended_action"]
        ),
    }


def _auto_validate(itinerary: dict, pace: str, distances: dict) -> dict:
    """Validate an itinerary without LLM calls."""
    from config.settings import get_settings
    s = get_settings()

    errors = []
    warnings = []
    max_km = {
        "relaxed": s.validation_max_km_relaxed,
        "moderate": s.validation_max_km_moderate,
        "intensive": s.validation_max_km_intensive,
    }.get(pace, s.validation_max_km_moderate)
    min_acts = 2
    max_acts = {
        "relaxed": s.validation_max_acts_relaxed,
        "moderate": s.validation_max_acts_moderate,
        "intensive": s.validation_max_acts_intensive,
    }.get(pace, s.validation_max_acts_moderate)
    max_driving_hours = s.validation_max_driving_hours

    days = itinerary.get("days", [])
    all_normalized_names = []

    for day in days:
        day_num = day.get("day", "?")
        acts = day.get("activities", [])
        km = day.get("total_distance_km", 0)
        hours = day.get("total_driving_hours", 0)

        if len(acts) < min_acts:
            warnings.append(f"Day {day_num}: only {len(acts)} activity (min {min_acts})")

        if len(acts) > max_acts:
            errors.append(f"Day {day_num}: {len(acts)} activities > {max_acts} limit for {pace}")

        if km > max_km:
            errors.append(f"Day {day_num}: {km:.0f} km exceeds {max_km} km limit for {pace}")
        elif km > max_km * 0.85:
            warnings.append(f"Day {day_num}: {km:.0f} km is close to {max_km} km limit")

        if len(acts) > 1 and km == 0:
            warnings.append(
                f"Day {day_num}: {len(acts)} places but total_distance_km=0 "
                f"— distance data missing"
            )

        if hours > max_driving_hours:
            errors.append(f"Day {day_num}: {hours:.1f}h driving > {max_driving_hours}h daily limit")

        for a in acts:
            norm = _normalize_place_name(a.get("name", ""))
            if norm and norm in all_normalized_names:
                errors.append(
                    f"Day {day_num}: '{a.get('name')}' already appeared in a previous day"
                )
            all_normalized_names.append(norm)

    score = max(0.0, round(1.0 - len(errors) * 0.3 - len(warnings) * 0.1, 2))
    action = "retry" if errors else "proceed"

    return {
        "is_valid": len(errors) == 0,
        "score": score,
        "errors": errors,
        "warnings": warnings,
        "recommended_action": action,
    }


def _normalize_place_name(name: str) -> str:
    """Normalize place names for duplicate detection."""
    return re.sub(r'[^a-zа-яё0-9]', '', name.lower())
