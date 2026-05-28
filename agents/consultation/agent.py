"""Consultation agent for pre-plan clarifications."""

from __future__ import annotations

import logging

from agents.base import make_scratchpad
from state.contracts import QueryClassification

logger = logging.getLogger(__name__)


def _format_questions(questions: list[str], language: str) -> str:
    if language.startswith("ru"):
        intro = "Чтобы составить хороший маршрут, уточните, пожалуйста:"
    else:
        intro = "To build a good itinerary, could you clarify:"
    return "\n".join([intro] + [f"- {q}" for q in questions])


async def consultation_agent_node(state: dict) -> dict:
    request_id = state.get("request_id", "unknown")
    query = state.get("user_query", "")
    language = state.get("user_language", "en")
    params = state.get("trip_parameters") or {}
    has_plan = bool(state.get("has_current_plan"))

    missing = []
    if not params.get("region"):
        missing.append("region")
    if not params.get("days") and "day" not in query.lower() and "дн" not in query.lower():
        missing.append("days")
    if not params.get("pace"):
        missing.append("pace")

    if missing and not has_plan:
        questions = []
        if "region" in missing:
            questions.append("Which region of Georgia do you want to focus on?")
        if "days" in missing:
            questions.append("How many days do you want to travel?")
        if "pace" in missing:
            questions.append("Do you prefer a relaxed, moderate, or intensive pace?")

        final_text = _format_questions(questions, language)
        classification = QueryClassification(
            intent="CONSULT",
            region=str(params.get("region", "")),
            search_query=str(params.get("search_query", query)),
            days=params.get("days"),
            pace=params.get("pace", "moderate"),
            should_ask_clarification=True,
            conversation_stage="CONSULT",
            reasoning="Missing planning parameters",
            steps=[{
                "agent": "consultation_agent",
                "params": {"missing": missing},
                "reason": "ask clarifying questions",
            }],
            estimated_agents=1,
        )
        logger.info(f"[{request_id}] consultation: asking {len(questions)} questions, missing={missing}")
        return {
            "conversation_stage": "CONSULT",
            "intent": "CONSULT",
            "orchestration": classification.model_dump(),
            "follow_up_questions": questions,
            "final_response": final_text,
            "task_complete": True,
            "agent_history": ["consultation_agent"],
            "agent_scratchpad": make_scratchpad("consultation_agent", f"ASKED: {questions}"),
            "router_trace": [{
                "from": "consultation_agent",
                "next": "memory_save",
                "reason": "clarification questions returned to user",
            }],
        }

    classification = QueryClassification(
        intent="PLAN",
        region=str(params.get("region", "")),
        search_query=str(params.get("search_query", query)),
        days=params.get("days"),
        pace=params.get("pace", "moderate"),
        should_ask_clarification=False,
        conversation_stage="PLAN",
        reasoning="Consultation complete; enough info to plan",
        steps=[
            {"agent": "search_agent", "params": {"region": str(params.get("region", "")), "search_query": str(params.get("search_query", query))}, "reason": "proceed to plan"},
            {"agent": "geo_agent", "params": {}, "reason": "geocode for distances"},
            {"agent": "planning_agent", "params": {"days": params.get("days"), "pace": params.get("pace", "moderate")}, "reason": "build itinerary"},
            {"agent": "validation_agent", "params": {}, "reason": "validate"},
            {"agent": "response_agent", "params": {}, "reason": "format final"},
        ],
        estimated_agents=5,
    )
    logger.info(f"[{request_id}] consultation: enough info, proceeding to PLAN")
    return {
        "conversation_stage": "PLAN",
        "intent": "PLAN",
        "orchestration": classification.model_dump(),
        "follow_up_questions": [],
        "agent_history": ["consultation_agent"],
        "agent_scratchpad": make_scratchpad("consultation_agent", "ENOUGH INFO -> PLAN"),
    }
