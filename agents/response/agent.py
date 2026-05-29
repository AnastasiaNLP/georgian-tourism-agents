# agents/response/agent.py
"""
ResponseAgent performs one final LLM formatting call.

It does not use tools. It formats either an itinerary or search results in the
user's language.

Node wrapper: response_agent_node(state: dict) → dict
"""
from __future__ import annotations
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.base import make_scratchpad
from agents.response.prompts import get_response_prompt

logger = logging.getLogger(__name__)


async def response_agent_node(state: dict) -> dict:
    """
    Format the final response with one LLM call.
    """
    from config.settings import get_settings

    request_id = state.get("request_id", "unknown")
    logger.info(f"[{request_id}] response_agent: start")

    user_query = state.get("user_query", "")
    user_language = state.get("user_language", "en")
    execution_mode = state.get("execution_mode", "normal")
    enriched = state.get("enriched_itinerary")
    raw = state.get("raw_itinerary")
    search_results = state.get("search_results") or []

    itinerary = enriched or raw

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.4,
        max_tokens=2500,
        api_key=get_settings().openai_api_key,
    )

    system_prompt = get_response_prompt(language=user_language, mode=execution_mode)

    # Build LLM input content.
    import json
    content_parts = [f"USER REQUEST: {user_query}\n"]

    if itinerary:
        content_parts.append(
            f"ITINERARY DATA:\n```json\n"
            f"{json.dumps(itinerary, ensure_ascii=False, indent=2)}\n```"
        )
    elif search_results:
        places_text = "\n".join(
            f"- {p.get('name')} ({p.get('location')}, {p.get('category', '')}): "
            f"{p.get('description', '')[:120]}"
            for p in search_results[:8]
        )
        content_parts.append(f"SEARCH RESULTS:\n{places_text}")
    else:
        content_parts.append("No itinerary or search results available.")

    # Add validation feedback when available.
    val = state.get("validation_result") or {}
    if val.get("warnings"):
        content_parts.append(f"\nNOTES: {'; '.join(val['warnings'][:2])}")

    user_content = "\n\n".join(content_parts)

    from monitoring.token_tracker import TokenTracker, merge_budget
    tracker = TokenTracker(model="gpt-4o-mini")

    plan_fields: dict = {}
    if itinerary:
        plan_fields = {
            "has_current_plan": True,
            "current_plan": itinerary,
            "current_plan_status": "active",
            "current_plan_version": max(state.get("current_plan_version") or 0, 1),
        }

    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ], config={"callbacks": [tracker]})

        final_response = response.content
        logger.info(f"[{request_id}] response_agent: {len(final_response)} chars")

        return {
            "final_response": final_response,
            "task_complete": True,
            "agent_history": ["response_agent"],
            "budget_state": merge_budget(state.get("budget_state") or {}, tracker),
            "agent_scratchpad": make_scratchpad(
                agent="response_agent",
                summary=f"Generated response ({len(final_response)} chars)",
            ),
            **plan_fields,
        }

    except Exception as e:
        logger.error(f"[{request_id}] response_agent failed: {e}")
        fallback = _fallback_response(user_query, itinerary, search_results, user_language)
        return {
            "final_response": fallback,
            "task_complete": True,
            "agent_history": ["response_agent"],
            "errors": [f"response_agent: {str(e)}"],
            "agent_scratchpad": make_scratchpad("response_agent", "FALLBACK response"),
            **plan_fields,
        }


def _fallback_response(query, itinerary, search_results, language) -> str:
    if itinerary:
        days = itinerary.get("days", [])
        lines = [f"Travel plan ({itinerary.get('total_days')} days):"]
        for day in days:
            lines.append(f"\nDay {day.get('day')}: {day.get('location')}")
            for act in day.get("activities", []):
                lines.append(f"  • {act.get('name')}")
        return "\n".join(lines)
    elif search_results:
        lines = [f"Found {len(search_results)} places:"]
        for p in search_results[:5]:
            lines.append(f"• {p.get('name')} — {p.get('location')}")
        return "\n".join(lines)
    return f"Here is information about your trip: {query}"
