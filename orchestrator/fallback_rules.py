"""
Rule-based fallback for the orchestrator.

Used when the LLM call fails or returns a plan that violates hard constraints.
"""

import logging
from state.models import OrchestratorPlan, OrchestratorStep, OrchestratorDecision

logger = logging.getLogger(__name__)

# The LLM cannot route outside this agent set.
ALLOWED_AGENTS = {
    "search_agent",
    "planning_agent",
    "geo_agent",
    "validation_agent",
    "response_agent",
}


def create_fallback_plan(state: dict) -> OrchestratorPlan:
    """
    Create a deterministic fallback plan.
    """
    query = (state.get("user_query") or "").lower()
    region = _detect_region(query)

    logger.warning(f"Using fallback plan, region={region}")

    # Decide whether the request needs itinerary planning.
    planning_keywords = ["день", "дня", "дней", "day", "days", "план", "plan",
                         "маршрут", "route", "itinerary", "поездка", "trip"]
    needs_planning = any(kw in query for kw in planning_keywords)

    if needs_planning:
        steps = [
            OrchestratorStep(
                agent="search_agent",
                params={"region": region, "max_results": 10},
                reason="fallback: find places in detected region"
            ),
            OrchestratorStep(
                agent="geo_agent",
                params={},
                reason="fallback: geocode + distance matrix before planning"
            ),
            OrchestratorStep(
                agent="planning_agent",
                params={"days": _detect_days(query), "pace": "moderate"},
                reason="fallback: create itinerary"
            ),
            OrchestratorStep(
                agent="validation_agent",
                params={},
                reason="fallback: validate itinerary"
            ),
            OrchestratorStep(
                agent="response_agent",
                params={},
                reason="fallback: format response"
            ),
        ]
        reasoning = f"Fallback plan: planning trip in {region}"
    else:
        steps = [
            OrchestratorStep(
                agent="search_agent",
                params={"region": region, "max_results": 10},
                reason="fallback: search places"
            ),
            OrchestratorStep(
                agent="response_agent",
                params={},
                reason="fallback: format response"
            ),
        ]
        reasoning = f"Fallback plan: search in {region}"

    return OrchestratorPlan(
        steps=steps,
        reasoning=reasoning,
        estimated_agents=len(steps),
    )


def create_fallback_review(state: dict) -> OrchestratorDecision:
    """
    Create a deterministic fallback review decision.
    """
    history = state.get("agent_history") or []
    logger.warning(f"Using fallback review, history={history}")

    # If response already ran, the workflow is complete.
    if "response_agent" in history:
        return OrchestratorDecision(
            next_action="done",
            reasoning="fallback: response already generated"
        )

    # search → geo → planning → validation → response
    if "search_agent" in history and "geo_agent" not in history:
        return OrchestratorDecision(
            next_action="call_agent",
            agent="geo_agent",
            params={},
            reasoning="fallback: geo needed before planning"
        )

    if "geo_agent" in history and "planning_agent" not in history:
        geocoded_count = sum(1 for p in (state.get("enriched_places") or []) if p.get("lat"))
        total_count = len(state.get("search_results") or [])
        # Continue once at least half of places were geocoded.
        if geocoded_count >= max(1, total_count // 2):
            return OrchestratorDecision(
                next_action="call_agent",
                agent="planning_agent",
                params={},
                reasoning=f"fallback: geo done ({geocoded_count}/{total_count} geocoded)"
            )
        else:
            return OrchestratorDecision(
                next_action="call_agent",
                agent="planning_agent",
                params={},
                reasoning="fallback: geo done (partial), proceed to planning"
            )

    if "planning_agent" in history and "validation_agent" not in history:
        return OrchestratorDecision(
            next_action="call_agent",
            agent="validation_agent",
            params={},
            reasoning="fallback: validate itinerary"
        )

    if "validation_agent" in history:
        return OrchestratorDecision(
            next_action="done",
            agent="response_agent",
            reasoning="fallback: pipeline complete"
        )

    # Default to response.
    return OrchestratorDecision(
        next_action="respond",
        reasoning="fallback: default to response"
    )


def enforce_plan_rules(plan: OrchestratorPlan, state: dict) -> OrchestratorPlan:
    """
    Fix an LLM plan when it violates hard routing constraints.

    Hard constraints:
    1. Only allowed agents
    2. First agent is search_agent
    3. Last agent is response_agent
    4. No consecutive duplicate agents
    5. Maximum 7 steps
    """
    steps = list(plan.steps)
    changed = False

    # Remove invalid agents.
    valid_steps = [s for s in steps if s.agent in ALLOWED_AGENTS]
    if len(valid_steps) != len(steps):
        logger.warning("Removed invalid agents from plan")
        steps = valid_steps
        changed = True

    # The first step must be search.
    if not steps or steps[0].agent != "search_agent":
        query = state.get("user_query") or ""
        region = _detect_region(query.lower())
        steps.insert(0, OrchestratorStep(
            agent="search_agent",
            params={"region": region},
            reason="enforced: search must be first"
        ))
        changed = True

    # Planning requires geo enrichment and validation.
    has_planning = any(s.agent == "planning_agent" for s in steps)
    has_geo = any(s.agent == "geo_agent" for s in steps)
    has_validation = any(s.agent == "validation_agent" for s in steps)
    has_response = any(s.agent == "response_agent" for s in steps)

    if has_planning:
        planning_idx = next(i for i, s in enumerate(steps) if s.agent == "planning_agent")

        # Insert geo before planning when it is missing.
        if not has_geo:
            steps.insert(planning_idx, OrchestratorStep(
                agent="geo_agent",
                params={},
                reason="enforced: geo BEFORE planning (provides coordinates + distances)"
            ))
            changed = True
            logger.info("enforce_plan_rules: added geo_agent before planning")
        else:
            # Move geo before planning when needed.
            geo_idx = next(i for i, s in enumerate(steps) if s.agent == "geo_agent")
            planning_idx = next(i for i, s in enumerate(steps) if s.agent == "planning_agent")
            if geo_idx > planning_idx:
                geo_step = steps.pop(geo_idx)
                planning_idx = next(i for i, s in enumerate(steps) if s.agent == "planning_agent")
                steps.insert(planning_idx, geo_step)
                changed = True
                logger.info("enforce_plan_rules: moved geo_agent before planning")

        # Insert validation before response when it is missing.
        if not has_validation:
            response_idx = next(
                (i for i, s in enumerate(steps) if s.agent == "response_agent"),
                len(steps)
            )
            steps.insert(response_idx, OrchestratorStep(
                agent="validation_agent",
                params={},
                reason="enforced: validation required before response"
            ))
            changed = True
            logger.info("enforce_plan_rules: added validation_agent")

    # The last step must be response.
    if not steps or steps[-1].agent != "response_agent":
        steps.append(OrchestratorStep(
            agent="response_agent",
            params={},
            reason="enforced: response must be last"
        ))
        changed = True

    # Remove consecutive duplicates.
    deduped = [steps[0]]
    for s in steps[1:]:
        if s.agent != deduped[-1].agent:
            deduped.append(s)
        else:
            logger.warning(f"Removed consecutive duplicate: {s.agent}")
            changed = True
    steps = deduped

    # Limit the plan length.
    if len(steps) > 7:
        # Keep the first and last steps, trim the middle.
        steps = steps[:6] + [steps[-1]]
        changed = True
        logger.warning("Plan trimmed to 7 steps")

    if changed:
        logger.info(f"Plan adjusted by enforce_plan_rules: {[s.agent for s in steps]}")

    return OrchestratorPlan(
        steps=steps,
        reasoning=plan.reasoning,
        estimated_agents=len(steps),
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _detect_region(query: str) -> str:
    """
    Detect the Georgian region from query keywords.
    Reads region-to-keyword mapping from config/georgia_regions.json via load_regions().
    """
    from agents.geo_filter import load_regions

    regions = load_regions()
    query_lower = query.lower()
    for region, keywords in regions.items():
        if any(kw.lower() in query_lower for kw in keywords):
            return region
    return "Tbilisi"  # default


def _detect_days(query: str) -> int:
    """Detect requested trip duration in days."""
    import re
    # Match numeric duration patterns in English or Russian.
    patterns = [
        r'(\d+)\s*(?:дня|дней|день|day|days)',
        r'(?:на|for)\s*(\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            days = int(match.group(1))
            return max(1, min(days, 14))
    return 3  # default
