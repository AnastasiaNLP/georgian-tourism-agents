# agents/validation/agent.py
"""
ValidationAgent performs programmatic itinerary checks.
The LLM is used only for warnings when explicitly enabled.

Node wrapper: validation_agent_node(state: dict) → dict
"""
from __future__ import annotations
import logging
import math
import re

from agents.base import extract_pace, make_scratchpad, route_key

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
    history = state.get("agent_history") or []
    planning_count = history.count("planning_agent")

    # Only trust geo data produced in THIS turn. enriched_places and distance_matrix
    # persist in the checkpoint across turns; on a revision turn (geo step skipped)
    # they are stale and must not drive distance checks. agent_history is reset every
    # turn, so it reliably signals whether the geo step ran now.
    geo_ran = "geo_agent" in history
    distances = (state.get("distance_matrix") or {}) if geo_ran else {}
    # Ground-truth coordinates from the geo step let us check that the itinerary's
    # reported distances are physically plausible, instead of trusting the planner's
    # self-reported numbers.
    coords_by_name = _coords_by_name(state.get("enriched_places") or []) if geo_ran else {}

    # First pass: deterministic validation.
    result = _auto_validate(itinerary, pace, distances, coords_by_name)

    # Hard errors (including reported distances that fall below the measured route)
    # do not need LLM review — re-plan, or proceed degraded once retries are spent
    # or the time/cost budget is exhausted (re-planning would only spend more).
    if result["errors"]:
        is_degraded = planning_count >= 2 or _budget_exhausted(state)
        if is_degraded:
            result["recommended_action"] = "proceed"
        logger.info(
            f"[{request_id}] validation_agent: FAILED programmatic, "
            f"{len(result['errors'])} errors"
            f"{' (stopping → degraded)' if is_degraded else ' (→ retry)'}"
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

    # Too few legs could be measured (e.g. geocoding/routing failures). Re-planning
    # will not recover the missing data, so proceed in degraded mode and let the
    # response disclose the reduced confidence to the user.
    if result.get("low_distance_confidence"):
        logger.info(
            f"[{request_id}] validation_agent: distance grounding below threshold "
            f"→ degraded (proceed, no retry)"
        )
        return {
            "validation_result": result,
            "execution_mode": "degraded",
            "errors": ["validation: route distances could not be verified — geo data incomplete"],
            "agent_history": ["validation_agent"],
            "agent_scratchpad": make_scratchpad(
                "validation_agent",
                f"DEGRADED: low distance grounding (score={result['score']})",
                is_valid=result["is_valid"], action="proceed"
            ),
        }

    if not result["warnings"]:
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


def _auto_validate(itinerary: dict, pace: str, distances: dict,
                   coords_by_name: dict | None = None) -> dict:
    """Validate an itinerary without LLM calls."""
    from config.settings import get_settings
    s = get_settings()

    coords_by_name = coords_by_name or {}

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
    tolerance = s.validation_distance_tolerance

    days = itinerary.get("days", [])
    all_normalized_names = []
    total_legs = 0
    covered_legs = 0

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

        # Cross-check the reported distance against the measured route. The measured
        # value is a lower bound (driving distance >= straight line), so a reported
        # total falling below it past the tolerance means the planner under-counted.
        measured_km, covered, legs = _measure_day_distance(acts, distances, coords_by_name)
        total_legs += legs
        covered_legs += covered
        if covered > 0 and measured_km > 0 and km < measured_km * (1 - tolerance):
            errors.append(
                f"Day {day_num}: reported {km:.0f} km but measured route is "
                f"~{measured_km:.0f} km"
            )

        for a in acts:
            norm = _normalize_place_name(a.get("name", ""))
            if norm and norm in all_normalized_names:
                errors.append(
                    f"Day {day_num}: '{a.get('name')}' already appeared in a previous day"
                )
            all_normalized_names.append(norm)

    # Distance grounding: only meaningful when geo data exists this turn. On flows
    # where the geo step is skipped (no matrix and no coordinates) we cannot judge
    # grounding and must not flag low confidence.
    has_ground_truth = bool(distances) or bool(coords_by_name)
    grounding = (covered_legs / total_legs) if total_legs else 1.0
    low_distance_confidence = bool(
        has_ground_truth and total_legs > 0 and grounding < s.validation_min_grounding
    )

    score = max(0.0, round(1.0 - len(errors) * 0.3 - len(warnings) * 0.1, 2))
    action = "retry" if errors else "proceed"

    return {
        "is_valid": len(errors) == 0,
        "score": score,
        "errors": errors,
        "warnings": warnings,
        "recommended_action": action,
        "low_distance_confidence": low_distance_confidence,
        "distance_grounding": round(grounding, 2),
    }


def _budget_exhausted(state: dict) -> bool:
    """Whether the request's wall-time or cost budget is spent (limits from settings)."""
    from config.settings import get_settings

    s = get_settings()
    budget = state.get("budget_state") or {}
    if budget.get("estimated_cost_usd", 0.0) >= s.execution_max_cost_usd:
        return True

    start = state.get("execution_start_time")
    if start:
        from datetime import datetime, timezone

        if isinstance(start, (int, float)):
            start = datetime.fromtimestamp(start, tz=timezone.utc)
        elif isinstance(start, str):
            start = datetime.fromisoformat(start)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        if elapsed >= s.execution_max_wall_time_seconds:
            return True
    return False


def _coords_by_name(enriched_places: list) -> dict:
    """Map normalized place name → (lat, lon) for places geocoded by the geo step."""
    coords = {}
    for p in enriched_places:
        lat, lon = p.get("lat"), p.get("lon")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            norm = _normalize_place_name(p.get("name", ""))
            if norm:
                coords[norm] = (float(lat), float(lon))
    return coords


def _matrix_lookup(distances: dict, name_a: str, name_b: str) -> float | None:
    """Return measured driving km between two places from the geo distance matrix."""
    entry = distances.get(route_key(name_a, name_b)) or distances.get(route_key(name_b, name_a))
    if isinstance(entry, dict) and entry.get("km", 0) > 0:
        return float(entry["km"])
    return None


def _measure_day_distance(acts: list, distances: dict, coords_by_name: dict) -> tuple:
    """Measure a day's route length from ground-truth data.

    For each consecutive pair, prefer the measured driving distance from the geo
    matrix; fall back to the straight-line distance between geocoded coordinates
    (a physical lower bound). Pairs with neither source are left uncovered.

    Returns (measured_km, covered_legs, total_legs).
    """
    total_legs = 0
    covered_legs = 0
    measured_km = 0.0
    for a, b in zip(acts, acts[1:]):
        total_legs += 1
        name_a = a.get("name", "")
        name_b = b.get("name", "")

        leg = _matrix_lookup(distances, name_a, name_b)
        if leg is not None:
            measured_km += leg
            covered_legs += 1
            continue

        ca = coords_by_name.get(_normalize_place_name(name_a))
        cb = coords_by_name.get(_normalize_place_name(name_b))
        if ca and cb:
            measured_km += _haversine_km(ca[0], ca[1], cb[0], cb[1])
            covered_legs += 1

    return measured_km, covered_legs, total_legs


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two points."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def _normalize_place_name(name: str) -> str:
    """Normalize place names for duplicate detection."""
    return re.sub(r'[^a-zа-яё0-9]', '', name.lower())
