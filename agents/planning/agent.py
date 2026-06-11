"""
PlanningAgent creates an itinerary with one LLM call.

Node wrapper: planning_agent_node(state: dict) → dict
"""
from __future__ import annotations
import json
import logging
import re

from agents.base import extract_trip_days, extract_pace, make_scratchpad

logger = logging.getLogger(__name__)


async def planning_agent_node(state: dict) -> dict:
    """Create an itinerary from GeoAgent-enriched places."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    from monitoring.token_tracker import TokenTracker, merge_budget
    from config.settings import get_settings

    request_id = state.get("request_id", "unknown")

    # Prefer GeoAgent output; fall back to raw search results.
    places = state.get("enriched_places") or state.get("search_results") or []
    distances = state.get("distance_matrix") or {}
    user_query = state.get("user_query", "")
    days = extract_trip_days(state)
    pace = extract_pace(state)
    region = (
        (state.get("search_context") or {}).get("region")
        or (state.get("trip_parameters") or {}).get("region")
        or "Georgia"
    )
    history = state.get("agent_history") or []
    iteration = history.count("planning_agent")

    if not places:
        return {
            "errors": ["planning_agent: no places available"],
            "agent_history": ["planning_agent"],
            "agent_scratchpad": make_scratchpad("planning_agent", "SKIPPED: no places"),
        }

    places_block = _format_rich_places(places, distances)
    max_km = {"relaxed": 100, "moderate": 150, "intensive": 200}.get(pace, 150)
    max_acts = {"relaxed": 3, "moderate": 4, "intensive": 6}.get(pace, 4)

    retry_note = ""
    if iteration > 0:
        errors = state.get("errors") or []
        retry_note = f"\n\nPREVIOUS ATTEMPT ISSUES: {errors[-2:]}. Make a simpler plan."

    prompt = f"""Build a {days}-day travel itinerary.

REQUEST: {user_query}
REGION: {region} | PACE: {pace} | MAX: {max_km} km/day, {max_acts} activities/day

{places_block}

RULES:
- ONLY use places from the list — never invent places
- Group nearby places (low km between them) on the same day
- If no direct A→C distance, sum A→B + B→C
- Copy coordinates exactly from the data for each activity
- Each day: 2–{max_acts} activities (never 1, never more than {max_acts})
- Remote places (>60 km from base) need a full day allocated
- total_distance_km must include RETURN trips for out-and-back destinations{retry_note}

OUTPUT — ONLY valid JSON, no markdown, no explanation:
{{"days": [{{"day": 1, "location": "Area name", "activities": [{{"name": "...",
"location": "...", "category": "...", "coordinates": {{"lat": 0.0, "lon": 0.0}},
"description": "..."}}], "total_distance_km": 0, "total_driving_hours": 0.0}}],
"total_days": {days}, "pace": "{pace}", "summary": "..."}}"""

    llm = ChatOpenAI(
        model="gpt-4o-mini", temperature=0.2, max_tokens=3000,
        api_key=get_settings().openai_api_key,
        timeout=60,
    )
    tracker = TokenTracker(model="gpt-4o-mini")

    try:
        response = await llm.ainvoke(
            [HumanMessage(content=prompt)],
            config={"callbacks": [tracker]},
        )
        raw = _try_parse_json(response.content, pace, days)
        if not raw:
            logger.warning(f"[{request_id}] planning_agent: JSON parse failed, using fallback")
            raw = _fallback_itinerary(places, days, pace, region, distances)

        activities_total = sum(len(d.get("activities", [])) for d in raw.get("days", []))

        return {
            "raw_itinerary": raw,
            "trip_parameters": {"days": days, "pace": pace},
            "agent_history": ["planning_agent"],
            "budget_state": merge_budget(state.get("budget_state") or {}, tracker),
            "agent_scratchpad": make_scratchpad(
                "planning_agent",
                f"Created {raw.get('total_days')}-day plan, {activities_total} activities (1 LLM call)",
                days=raw.get("total_days"),
                pace=pace,
            ),
        }

    except Exception as e:
        logger.error(f"[{request_id}] planning_agent failed: {e}")
        fallback = _fallback_itinerary(places, days, pace, region, distances)
        return {
            "raw_itinerary": fallback,
            "trip_parameters": {"days": days, "pace": pace},
            "agent_history": ["planning_agent"],
            "errors": [f"planning_agent: {str(e)[:100]}"],
            "budget_state": merge_budget(state.get("budget_state") or {}, tracker),
            "agent_scratchpad": make_scratchpad("planning_agent", f"FALLBACK: {str(e)[:80]}"),
        }


def _format_rich_places(places: list, distances: dict) -> str:
    """
    Format enriched places and distance matrix for the LLM prompt.
    """
    lines = ["AVAILABLE PLACES (with coordinates and distances):"]

    for i, p in enumerate(places[:20]):
        lat = p.get("lat", "?")
        lon = p.get("lon", "?")
        coord_str = f"lat={lat}, lon={lon}" if lat != "?" else "coordinates: unknown"
        tags_str = ", ".join(p.get("tags") or []) or "—"
        desc = (p.get("description") or "")[:150]

        lines.append(
            f"\n[{i+1}] {p.get('name', 'Unknown')}\n"
            f"  Location: {p.get('location', 'Georgia')} | {coord_str}\n"
            f"  Category: {p.get('category', '—')} | Tags: {tags_str}\n"
            f"  Description: {desc}"
        )

    if distances:
        lines.append("\nDISTANCES (driving):")
        for route, data in list(distances.items())[:30]:
            lines.append(f"  {route}: {data.get('km', 0):.0f} km / {data.get('hours', 0):.1f}h")

    return "\n".join(lines)


def _fallback_itinerary(places: list, days: int, pace: str,
                         region: str, distances: dict = None) -> dict:
    """Fallback planner: group by proximity when distances are available."""
    if distances:
        grouped = _group_by_proximity(places, distances, days, pace)
    else:
        grouped = _simple_split(places, days, pace, region)

    return {
        "days": grouped,
        "total_days": len(grouped),
        "pace": pace,
        "summary": f"Fallback {len(grouped)}-day trip in {region}",
    }


def _group_by_proximity(places: list, distances: dict, days: int, pace: str) -> list:
    """
    Group places by proximity using a greedy nearest-neighbor heuristic.
    """
    per_day = {"relaxed": 2, "moderate": 3, "intensive": 4}.get(pace, 3)
    remaining = list(places)
    days_list = []
    day_sizes = _target_day_sizes(len(remaining), days, per_day)

    for day_num, target_size in enumerate(day_sizes, start=1):
        if not remaining:
            break

        day_places = [remaining.pop(0)]

        while len(day_places) < target_size and remaining:
            last = day_places[-1]
            last_name = last.get("name", "")[:25]

            best_candidate = None
            best_km = float("inf")

            for candidate in remaining:
                cand_name = candidate.get("name", "")[:25]
                key = f"{last_name}→{cand_name}"
                rev_key = f"{cand_name}→{last_name}"
                km = distances.get(key, distances.get(rev_key, {})).get("km", 999)
                if km < best_km:
                    best_km = km
                    best_candidate = candidate

            if best_candidate:
                remaining.remove(best_candidate)
                day_places.append(best_candidate)
            else:
                break

        region_name = day_places[0].get("location", "Georgia").split(",")[0].strip()
        days_list.append({
            "day": day_num,
            "location": region_name,
            "activities": [{
                "name": p.get("name", ""),
                "location": p.get("location", ""),
                "category": p.get("category", "attraction"),
                "coordinates": {"lat": p.get("lat"), "lon": p.get("lon")},
                "description": (p.get("description") or "")[:100],
            } for p in day_places],
            "total_distance_km": 0,
            "total_driving_hours": 0,
        })

    return days_list or [{"day": 1, "location": "Georgia", "activities": []}]


def _simple_split(places: list, days: int, pace: str, region: str) -> list:
    """Simple split when distance data is unavailable."""
    per_day = {"relaxed": 2, "moderate": 3, "intensive": 4}.get(pace, 3)
    days_list = []
    offset = 0
    for i, size in enumerate(_target_day_sizes(len(places), days, per_day)):
        chunk = places[offset:offset + size]
        offset += size
        if not chunk:
            break
        days_list.append({
            "day": i + 1,
            "location": chunk[0].get("location", region).split(",")[0].strip(),
            "activities": [{
                "name": p.get("name", ""),
                "location": p.get("location", region),
                "category": p.get("category", "attraction"),
                "coordinates": {"lat": p.get("lat"), "lon": p.get("lon")},
                "description": (p.get("description") or "")[:100],
            } for p in chunk],
            "total_distance_km": 0,
            "total_driving_hours": 0,
        })
    return days_list or [{"day": 1, "location": region, "activities": []}]


def _target_day_sizes(total_places: int, requested_days: int, max_per_day: int) -> list[int]:
    """
    Distribute places across requested days while respecting a per-day cap.

    The fallback planner should preserve the user's requested duration when
    there are enough places, instead of filling the first day to capacity.
    """
    if total_places <= 0:
        return []

    day_count = max(1, min(requested_days, total_places))
    sizes = [0] * day_count
    remaining = min(total_places, day_count * max_per_day)

    index = 0
    while remaining > 0:
        if sizes[index] < max_per_day:
            sizes[index] += 1
            remaining -= 1
        index = (index + 1) % day_count

    return sizes


def _try_parse_json(text: str, pace: str, days: int) -> dict:
    """Try to extract valid itinerary JSON from model output."""
    # 1. ```json ... ``` block
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        result = _parse_candidate(m.group(1).strip(), pace, days)
        if result:
            return result

    # 2. Whole text as JSON
    result = _parse_candidate(text.strip(), pace, days)
    if result:
        return result

    # 3. Search for a JSON object block.
    m = re.search(r'(\{.*\})', text, re.DOTALL)
    if m:
        result = _parse_candidate(m.group(1), pace, days)
        if result:
            return result

    return None


def _parse_candidate(text: str, pace: str, days: int) -> dict:
    """Parse a JSON candidate and validate the expected structure."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "days" in parsed:
            parsed["total_days"] = len(parsed["days"])
            parsed.setdefault("pace", pace)
            parsed.setdefault("summary", f"{days}-day trip")
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return None
