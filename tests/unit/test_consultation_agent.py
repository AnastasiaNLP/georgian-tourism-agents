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


@pytest.mark.asyncio
async def test_consultation_questions_are_russian_for_ru_language():
    """All three question strings must be in Russian when language='ru'."""
    from agents.consultation.agent import consultation_agent_node

    result = await consultation_agent_node({
        "request_id": "c4",
        "user_query": "хочу путешествовать",
        "user_language": "ru",
        "trip_parameters": {},
        "has_current_plan": False,
    })

    response = result["final_response"]
    # Russian intro must be present
    assert "уточните" in response
    # No English question fragments should appear
    assert "Which region" not in response
    assert "How many days" not in response
    assert "Do you prefer" not in response
    # Russian translations must appear
    assert "регион" in response
    assert "дней" in response or "дн" in response
    assert "темп" in response


@pytest.mark.asyncio
async def test_consultation_questions_are_english_for_en_language():
    """All three question strings must be in English when language='en'."""
    from agents.consultation.agent import consultation_agent_node

    result = await consultation_agent_node({
        "request_id": "c5",
        "user_query": "I want to travel",
        "user_language": "en",
        "trip_parameters": {},
        "has_current_plan": False,
    })

    response = result["final_response"]
    assert "clarify" in response
    assert "Чтобы" not in response
    assert "Which region" in response


@pytest.mark.asyncio
async def test_consultation_single_missing_param_asks_one_question():
    """Only missing params generate questions — region present → no region question."""
    from agents.consultation.agent import consultation_agent_node

    result = await consultation_agent_node({
        "request_id": "c6",
        "user_query": "I want to travel for 3 days",
        "user_language": "en",
        "trip_parameters": {"region": "Kakheti", "days": 3},
        "has_current_plan": False,
    })

    # Only pace is missing
    questions = result.get("follow_up_questions", [])
    assert len(questions) == 1
    assert "pace" in questions[0].lower() or "relaxed" in questions[0].lower() or "intensive" in questions[0].lower()
