from __future__ import annotations

import json
from typing import Any

from langchain_openai import ChatOpenAI

from agents.revision.prompts import get_revision_prompt
from monitoring.token_tracker import TokenTracker, merge_budget
from state.contracts import Itinerary, QueryClassification
from state.schema import TravelPlanningState


def _get_value(state: Any, key: str, default: Any = None) -> Any:
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _build_scratchpad(state: Any, status: str, message: str) -> dict[str, Any]:
    return {
        "agent": "revision_agent",
        "status": status,
        "message": message,
        "request_id": _get_value(state, "request_id", ""),
        "current_plan_version": _get_value(state, "current_plan_version", 0),
    }


def _build_success_orchestration(revised_plan: dict[str, Any], state: Any) -> dict[str, Any]:
    trip_parameters = _get_value(state, "trip_parameters", {})
    if isinstance(trip_parameters, dict):
        region = trip_parameters.get("region", "")
        search_query = trip_parameters.get("search_query", "")
    else:
        region = getattr(trip_parameters, "region", "")
        search_query = getattr(trip_parameters, "search_query", "")

    return QueryClassification(
        intent="REVISE",
        region=region,
        search_query=search_query,
        days=len(revised_plan.get("days", [])),
        pace=revised_plan.get("pace", "moderate"),
        should_ask_clarification=False,
        conversation_stage="REVISE",
        steps=[
            {"agent": "revision_agent"},
            {"agent": "validation_agent"},
            {"agent": "response_agent"},
        ],
        reasoning="Revision completed successfully; sending updated itinerary to validation.",
        estimated_agents=3,
    ).model_dump()


def _merge_budget(state: Any, tracker: TokenTracker | None) -> dict[str, Any]:
    budget_state = _get_value(state, "budget_state", {}) or {}
    if tracker is None:
        return budget_state
    return merge_budget(budget_state, tracker)


async def revision_agent_node(state: TravelPlanningState) -> dict[str, Any]:
    current_plan = _as_dict(_get_value(state, "current_plan")) or _as_dict(_get_value(state, "raw_itinerary"))
    feedback = _get_value(state, "feedback", "") or ""
    trip_parameters = _get_value(state, "trip_parameters", {})

    if not current_plan or not current_plan.get("days"):
        return {
            "conversation_stage": "REVISE",
            "intent": "REVISE",
            "has_current_plan": bool(current_plan),
            "current_plan_status": "revision_failed_no_plan",
            "current_plan": current_plan,
            "raw_itinerary": current_plan,
            "enriched_itinerary": {},
            "errors": ["revision_agent: no plan to revise"],
            "router_trace": [
                {
                    "from": "revision_agent",
                    "next": "validation_agent",
                    "reason": "no current plan available",
                }
            ],
            "agent_history": ["revision_agent"],
            "agent_scratchpad": _build_scratchpad(state, "skipped", "no current plan available"),
            "budget_state": _get_value(state, "budget_state", {}) or {},
            "task_complete": False,
        }

    prompt = get_revision_prompt()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(Itinerary)
    chain = prompt | structured_llm
    tracker = TokenTracker(model="gpt-4o-mini")

    try:
        result = await chain.ainvoke(
            {
                "feedback": feedback,
                "current_plan": json.dumps(current_plan, ensure_ascii=False, indent=2),
                "trip_parameters": json.dumps(_as_dict(trip_parameters), ensure_ascii=False, indent=2),
            },
            config={"callbacks": [tracker]},
        )
        if hasattr(result, "model_dump"):
            revised_plan = result.model_dump()
        elif isinstance(result, dict):
            revised_plan = result
        else:
            revised_plan = {}

        revised_plan.setdefault("days", [])
        revised_plan["total_days"] = len(revised_plan.get("days", []))
        current_plan_version = int(_get_value(state, "current_plan_version", 0) or 0) + 1

        scratchpad = _build_scratchpad(state, "success", "revision completed")
        scratchpad["new_plan_days"] = len(revised_plan.get("days", []))

        return {
            "conversation_stage": "REVISE",
            "intent": "REVISE",
            "has_current_plan": True,
            "current_plan_status": "revised",
            "current_plan_version": current_plan_version,
            "current_plan": revised_plan,
            "raw_itinerary": revised_plan,
            "enriched_itinerary": {},
            "final_response": "",
            "orchestration": _build_success_orchestration(revised_plan, state),
            "router_trace": [
                {
                    "from": "revision_agent",
                    "next": "validation_agent",
                    "reason": "structured revised itinerary ready",
                }
            ],
            "agent_history": ["revision_agent"],
            "agent_scratchpad": scratchpad,
            "budget_state": _merge_budget(state, tracker),
            "task_complete": False,
        }
    except Exception as exc:
        return {
            "conversation_stage": "REVISE",
            "intent": "REVISE",
            "has_current_plan": True,
            "current_plan_status": "revision_failed",
            "current_plan": current_plan,
            "raw_itinerary": current_plan,
            "enriched_itinerary": {},
            "errors": [f"revision_agent: {exc}"],
            "final_response": "",
            "orchestration": _build_success_orchestration(current_plan, state),
            "router_trace": [
                {
                    "from": "revision_agent",
                    "next": "validation_agent",
                    "reason": "kept original itinerary after failure",
                }
            ],
            "agent_history": ["revision_agent"],
            "agent_scratchpad": _build_scratchpad(state, "failed", str(exc)),
            "budget_state": _merge_budget(state, tracker),
            "task_complete": False,
        }


revision_agent = revision_agent_node
build_revision_agent = revision_agent_node

