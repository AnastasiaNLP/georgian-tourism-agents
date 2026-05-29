"""Unit tests for response_agent plan-persistence logic."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest


ITINERARY = {
    "days": [
        {"day": 1, "location": "Tbilisi", "activities": [{"name": "Old Town", "location": "Tbilisi", "category": "culture"}]},
        {"day": 2, "location": "Mtskheta", "activities": [{"name": "Jvari Monastery", "location": "Mtskheta", "category": "culture"}]},
    ],
    "total_days": 2,
    "pace": "moderate",
    "summary": "Tbilisi and Mtskheta",
}

BASE_STATE = {
    "request_id": "r1",
    "user_query": "2 days Georgia",
    "user_language": "en",
    "execution_mode": "normal",
    "raw_itinerary": ITINERARY,
    "enriched_itinerary": {},
    "search_results": [],
    "validation_result": {},
    "budget_state": {},
    "current_plan_version": 0,
}


def _mock_llm_response(text: str):
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content=text))
    return mock_llm


@pytest.mark.asyncio
async def test_response_agent_sets_current_plan_on_success():
    from agents.response.agent import response_agent_node

    with patch("agents.response.agent.ChatOpenAI", return_value=_mock_llm_response("Great plan!")):
        result = await response_agent_node(BASE_STATE)

    assert result["has_current_plan"] is True
    assert result["current_plan"] == ITINERARY
    assert result["current_plan_status"] == "active"
    assert result["current_plan_version"] == 1
    assert result["task_complete"] is True


@pytest.mark.asyncio
async def test_response_agent_version_starts_at_1_for_fresh_plan():
    from agents.response.agent import response_agent_node

    state = {**BASE_STATE, "current_plan_version": 0}
    with patch("agents.response.agent.ChatOpenAI", return_value=_mock_llm_response("Plan!")):
        result = await response_agent_node(state)

    assert result["current_plan_version"] == 1


@pytest.mark.asyncio
async def test_response_agent_preserves_higher_version_after_revision():
    from agents.response.agent import response_agent_node

    state = {**BASE_STATE, "current_plan_version": 3}
    with patch("agents.response.agent.ChatOpenAI", return_value=_mock_llm_response("Revised!")):
        result = await response_agent_node(state)

    assert result["current_plan_version"] == 3


@pytest.mark.asyncio
async def test_response_agent_no_plan_fields_without_itinerary():
    from agents.response.agent import response_agent_node

    state = {**BASE_STATE, "raw_itinerary": {}, "enriched_itinerary": {}}
    with patch("agents.response.agent.ChatOpenAI", return_value=_mock_llm_response("No itinerary.")):
        result = await response_agent_node(state)

    assert "has_current_plan" not in result
    assert "current_plan" not in result


@pytest.mark.asyncio
async def test_response_agent_sets_current_plan_on_fallback():
    from agents.response.agent import response_agent_node

    with patch("agents.response.agent.ChatOpenAI") as mock_cls:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))
        mock_cls.return_value = mock_llm
        result = await response_agent_node(BASE_STATE)

    assert result["has_current_plan"] is True
    assert result["current_plan"] == ITINERARY
    assert result["task_complete"] is True
    assert result["current_plan_version"] == 1
