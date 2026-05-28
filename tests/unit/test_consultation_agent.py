"""Unit tests for the consultation agent node."""

import pytest


@pytest.mark.asyncio
async def test_consultation_asks_questions_when_missing():
    from agents.consultation.agent import consultation_agent_node

    result = await consultation_agent_node({
        "request_id": "c1",
        "user_query": "I want to travel",
        "user_language": "en",
        "trip_parameters": {},
        "has_current_plan": False,
    })

    assert result["task_complete"] is True
    assert result["follow_up_questions"]
    assert result["final_response"]
    assert result["conversation_stage"] == "CONSULT"
    assert result["intent"] == "CONSULT"


@pytest.mark.asyncio
async def test_consultation_proceeds_when_enough_info():
    from agents.consultation.agent import consultation_agent_node

    result = await consultation_agent_node({
        "request_id": "c2",
        "user_query": "Kakheti 3 days relaxed",
        "user_language": "en",
        "trip_parameters": {"region": "Kakheti", "days": 3, "pace": "relaxed"},
        "has_current_plan": False,
    })

    assert result["conversation_stage"] == "PLAN"
    assert result["intent"] == "PLAN"
    assert "final_response" not in result
    assert result["follow_up_questions"] == []


@pytest.mark.asyncio
async def test_consultation_questions_formatted_for_russian():
    from agents.consultation.agent import consultation_agent_node

    result = await consultation_agent_node({
        "request_id": "c3",
        "user_query": "хочу путешествовать",
        "user_language": "ru",
        "trip_parameters": {},
        "has_current_plan": False,
    })

    assert result["task_complete"] is True
    assert "уточните" in result["final_response"]
