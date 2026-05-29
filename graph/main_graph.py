"""Main deterministic LangGraph pipeline."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from langgraph.graph import END, StateGraph

from agents.registry import get_registry
from orchestrator.execution_guard import ExecutionGuard
from orchestrator.orchestrator import OrchestratorAgent
from state.schema import TravelPlanningState

_orchestrator: Optional[OrchestratorAgent] = None
_guard = ExecutionGuard()

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_orchestrator() -> OrchestratorAgent:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = OrchestratorAgent()
    return _orchestrator


def _plan_to_dict(plan) -> dict:
    if hasattr(plan, "model_dump"):
        return plan.model_dump()
    if isinstance(plan, dict):
        return plan
    return {
        "steps": [
            step.model_dump() if hasattr(step, "model_dump") else dict(step)
            for step in getattr(plan, "steps", [])
        ],
        "reasoning": getattr(plan, "reasoning", ""),
        "estimated_agents": getattr(plan, "estimated_agents", 0),
    }


def _agent_from_step(step) -> str:
    return step.get("agent") if isinstance(step, dict) else step.agent


def _params_from_step(step) -> dict:
    return (step.get("params") if isinstance(step, dict) else step.params) or {}


def _merged_orchestrator_params(steps: list) -> dict:
    merged = {}
    for step in steps:
        params = _params_from_step(step)
        if params:
            merged.update(params)
    return merged


async def memory_load_node(state: TravelPlanningState) -> dict:
    request_id = state.get("request_id", "unknown")
    user_id = state.get("user_id")
    logger.info(f"[{request_id}] memory_load: initializing, user_id={user_id}")

    result = {
        "execution_start_time": datetime.now(timezone.utc),
        "execution_mode": "normal",
        "feature_flags": _load_feature_flags(),
        "task_complete": False,
        "budget_state": {
            "llm_calls": 0,
            "tool_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "estimated_cost_usd": 0.0,
        },
        # Per-turn fields: reset at the start of every turn so previous-turn
        # values do not leak into the new turn via the checkpointer.
        "agent_history": [],
        "errors": [],
        "router_trace": [],
        "eval_score": {},
        "final_response": "",
        "feedback": "",
        "follow_up_questions": [],
        "trip_parameters": {},
    }

    if user_id:
        try:
            from src.memory.memory_manager import get_memory_manager

            mm = get_memory_manager()
            profile = await mm.load_profile(user_id)
            result["user_memory"] = profile
            logger.info(f"[{request_id}] memory_load: profile loaded for {user_id}")
        except Exception as exc:
            logger.warning(f"[{request_id}] memory_load: failed to load profile: {exc}")
            result["user_memory"] = None
    else:
        result["user_memory"] = None

    return result


def _load_feature_flags() -> dict:
    from config.settings import get_settings

    settings = get_settings()
    return {
        "USE_WEATHER": settings.feature_use_weather,
        "ENABLE_RAG": settings.feature_enable_rag,
        "ENABLE_GEO_ROUTES": settings.feature_enable_geo_routes,
        "ENABLE_VALIDATION": settings.feature_enable_validation,
        "ENABLE_VALIDATION_LLM": False,
        "ENABLE_EVAL": settings.feature_enable_eval,
        "LLM_MODEL": settings.llm_model,
        "MAX_SEARCH_RESULTS": settings.max_search_results,
    }


async def orchestrator_plan_node(state: TravelPlanningState) -> dict:
    request_id = state.get("request_id", "unknown")

    guard_result = _guard.check_before_plan(state)
    if not guard_result.allowed:
        logger.warning(f"[{request_id}] Guard blocked plan: {guard_result.reason}")
        return {
            "orchestrator_plan": {"steps": [], "reasoning": guard_result.reason, "estimated_agents": 0},
            "orchestrator_decision": "response_agent",
            "orchestrator_params": {},
            "intent": "INFO",
            "conversation_stage": "INFO",
            "execution_mode": "degraded",
            "errors": [f"guard: {guard_result.reason}"],
            "router_trace": [{
                "from": "orchestrator_plan",
                "next": "response_agent",
                "reason": guard_result.reason,
                "ts": _now_iso(),
            }],
        }

    from monitoring.token_tracker import TokenTracker, merge_budget

    tracker = TokenTracker(model="claude-haiku-4-5-20251001")
    classification = await _get_orchestrator().classify(state, callbacks=[tracker])
    plan_dict = _plan_to_dict(classification)
    steps = plan_dict.get("steps") or []
    intent = classification.intent if hasattr(classification, "intent") else plan_dict.get("intent", "INFO")
    params = {
        "region": getattr(classification, "region", ""),
        "search_query": getattr(classification, "search_query", state.get("user_query", "")),
        "days": getattr(classification, "days", None),
        "pace": getattr(classification, "pace", "moderate"),
    }
    next_agent = {
        "PLAN": "search_agent",
        "INFO": "search_agent",
        "SEARCH": "search_agent",
        "CONSULT": "consultation_agent",
        "REVISE": "revision_agent",
    }.get(intent, "search_agent")

    logger.info(
        f"[{request_id}] classifier: intent={intent}, "
        f"params={params}, llm_plan={[_agent_from_step(step) for step in steps]}"
    )

    return {
        "orchestrator_plan": plan_dict,
        "orchestrator_decision": next_agent,
        "orchestrator_params": params,
        "intent": intent,
        "conversation_stage": getattr(classification, "conversation_stage", intent),
        "trip_parameters": {
            "days": params.get("days"),
            "pace": params.get("pace", "moderate"),
            "region": params.get("region", ""),
            "search_query": params.get("search_query", ""),
            "consent_to_plan": intent == "PLAN",
        },
        "orchestration": classification.model_dump() if hasattr(classification, "model_dump") else plan_dict,
        "budget_state": merge_budget(state.get("budget_state") or {}, tracker),
        "router_trace": [{
            "from": "classifier",
            "next": next_agent,
            "reason": f"deterministic {intent}: {(plan_dict.get('reasoning') or '')[:100]}",
            "ts": _now_iso(),
        }],
    }


def route_from_orchestrator(state: TravelPlanningState) -> str:
    if state.get("task_complete"):
        return "__end__"

    # Intent is the primary routing signal from the classifier.
    # Stage is derived UI state and only used as fallback.
    orchestration = state.get("orchestration") or {}
    intent = orchestration.get("intent") if isinstance(orchestration, dict) else getattr(orchestration, "intent", None)
    if intent is None:
        intent = state.get("intent")

    if intent == "CONSULT":
        return "consultation_agent"
    if intent == "REVISE":
        return "revision_agent"
    if intent in {"PLAN", "INFO", "SEARCH"}:
        return "search_agent"

    # Fallback to stage if intent is missing.
    stage = state.get("conversation_stage")
    if stage == "CONSULT":
        return "consultation_agent"
    if stage == "REVISE":
        return "revision_agent"
    if stage in {"PLAN", "INFO"}:
        return "search_agent"

    return _normalize_agent_decision(state.get("orchestrator_decision", "search_agent"))


def route_after_consultation(state: TravelPlanningState) -> str:
    if state.get("task_complete"):
        return "memory_save"
    orchestration = state.get("orchestration") or {}
    intent = orchestration.get("intent") if isinstance(orchestration, dict) else getattr(orchestration, "intent", None)
    if intent is None:
        intent = state.get("intent")
    if intent == "PLAN":
        return "search_agent"
    return "response_agent"


def route_after_search(state: TravelPlanningState) -> str:
    orchestration = state.get("orchestration") or {}
    intent = orchestration.get("intent") if isinstance(orchestration, dict) else getattr(orchestration, "intent", None)
    if intent is None:
        intent = state.get("intent")
    return "geo_agent" if intent == "PLAN" else "response_agent"


def route_after_planning(state: TravelPlanningState) -> str:
    flags = state.get("feature_flags") or {}
    return "validation_agent" if flags.get("ENABLE_VALIDATION", True) else "response_agent"


def _normalize_agent_decision(decision: Optional[str]) -> str:
    allowed = {
        "search_agent",
        "planning_agent",
        "geo_agent",
        "validation_agent",
        "response_agent",
        "consultation_agent",
        "revision_agent",
    }
    if decision not in allowed:
        return "response_agent"
    return decision


async def eval_node(state: TravelPlanningState) -> dict:
    """LLM-as-judge eval enabled through FEATURE_ENABLE_EVAL=true."""
    if not state.get("feature_flags", {}).get("ENABLE_EVAL", False):
        return {}

    itinerary = state.get("enriched_itinerary") or state.get("raw_itinerary")
    if not itinerary:
        return {}

    request_id = state.get("request_id", "unknown")
    logger.info(f"[{request_id}] eval_node: running LLM-as-judge")

    try:
        days = itinerary.get("days", [])
        total_km = sum(d.get("total_distance_km", 0) for d in days if isinstance(d, dict))
        pace = (state.get("trip_parameters") or {}).get("pace", "moderate")

        prompt = (
            f"Evaluate this Georgia travel itinerary:\n"
            f"- Days: {len(days)}, Total distance: {total_km:.0f} km, Pace: {pace}\n"
            f"Rate each criterion 0.0 to 1.0:\n"
            f"  realism: are the places real and worth visiting?\n"
            f"  distance: are daily distances reasonable for the pace?\n"
            f"  feasibility: can a tourist realistically complete this?\n"
            f"Respond ONLY with valid JSON, no markdown:\n"
            '{{"realism": 0.0, "distance": 0.0, "feasibility": 0.0, "comment": "brief"}}'
        )

        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=200)
        response = await llm.ainvoke(prompt)
        raw = response.content.strip()

        import json

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())

        total = round((parsed.get("realism", 0) + parsed.get("distance", 0) + parsed.get("feasibility", 0)) / 3, 3)
        parsed["total"] = total

        logger.info(f"[{request_id}] eval_node: score={total:.2f} comment={parsed.get('comment', '')[:50]}")
        return {"eval_score": parsed}
    except Exception as exc:
        logger.warning(f"[{request_id}] eval_node: failed: {exc}")
        return {"eval_score": {"total": 0.0, "comment": f"eval_failed: {exc}"}}


async def memory_save_node(state: TravelPlanningState) -> dict:
    request_id = state.get("request_id", "unknown")
    user_id = state.get("user_id")

    if not user_id:
        logger.info(f"[{request_id}] memory_save: no user_id, skipping")
        return {"memory_saved": False}

    try:
        from src.memory.memory_manager import get_memory_manager

        mm = get_memory_manager()
        profile_data = mm.extract_profile_update(state)
        if profile_data:
            await mm.save_profile(user_id, profile_data)
            logger.info(f"[{request_id}] memory_save: profile saved for {user_id}")

        mode = state.get("execution_mode", "normal")
        history = state.get("agent_history", [])
        has_itinerary = state.get("enriched_itinerary") or state.get("raw_itinerary")

        if mode == "normal" and "planning_agent" in history and has_itinerary:
            episode = mm.extract_episode(state)
            await mm.save_episode(user_id, episode)
            logger.info(f"[{request_id}] memory_save: episode saved for {user_id}")
        else:
            logger.info(
                f"[{request_id}] memory_save: skip episode "
                f"(mode={mode}, planning={'planning_agent' in history}, itinerary={bool(has_itinerary)})"
            )
        return {"memory_saved": True}
    except Exception as exc:
        logger.error(f"[{request_id}] memory_save: failed: {exc}")
        return {"memory_saved": False}


def build_graph(checkpointer=None):
    wf = StateGraph(TravelPlanningState)
    registry = get_registry()
    registry.initialize()

    wf.add_node("memory_load", memory_load_node)
    wf.add_node("orchestrator_plan", orchestrator_plan_node)
    wf.add_node("search_agent", registry.get("search_agent"))
    wf.add_node("geo_agent", registry.get("geo_agent"))
    wf.add_node("planning_agent", registry.get("planning_agent"))
    wf.add_node("validation_agent", registry.get("validation_agent"))
    wf.add_node("response_agent", registry.get("response_agent"))
    wf.add_node("consultation_agent", registry.get("consultation_agent"))
    wf.add_node("revision_agent", registry.get("revision_agent"))
    wf.add_node("eval_node", eval_node)
    wf.add_node("memory_save", memory_save_node)

    wf.set_entry_point("memory_load")
    wf.add_edge("memory_load", "orchestrator_plan")

    wf.add_edge("geo_agent", "planning_agent")
    wf.add_edge("validation_agent", "response_agent")
    wf.add_edge("revision_agent", "validation_agent")
    wf.add_edge("response_agent", "eval_node")
    wf.add_edge("eval_node", "memory_save")
    wf.add_edge("memory_save", END)

    wf.add_conditional_edges(
        "orchestrator_plan",
        route_from_orchestrator,
        {
            "search_agent": "search_agent",
            "planning_agent": "planning_agent",
            "geo_agent": "geo_agent",
            "validation_agent": "validation_agent",
            "response_agent": "response_agent",
            "consultation_agent": "consultation_agent",
            "revision_agent": "revision_agent",
            "__end__": END,
        },
    )

    wf.add_conditional_edges(
        "consultation_agent",
        route_after_consultation,
        {
            "search_agent": "search_agent",
            "response_agent": "response_agent",
            "memory_save": "memory_save",
        },
    )

    wf.add_conditional_edges(
        "search_agent",
        route_after_search,
        {
            "geo_agent": "geo_agent",
            "response_agent": "response_agent",
        },
    )

    wf.add_conditional_edges(
        "planning_agent",
        route_after_planning,
        {
            "validation_agent": "validation_agent",
            "response_agent": "response_agent",
        },
    )

    return wf.compile(checkpointer=checkpointer)


_cached_graph = None


async def run_graph(
    user_query: str,
    request_id: str,
    user_id: Optional[str] = None,
    user_language: Optional[str] = None,
    correlation_id: Optional[str] = None,
):
    from state.schema import create_initial_state

    global _cached_graph
    if _cached_graph is None:
        _cached_graph = build_graph()
    state = create_initial_state(
        user_query=user_query,
        request_id=request_id,
        correlation_id=correlation_id or request_id,
        user_id=user_id or "",
        user_language=user_language or "en",
    )
    config = {"configurable": {"thread_id": request_id}}
    return await _cached_graph.ainvoke(state, config=config)

