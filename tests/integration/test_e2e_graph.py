"""
E2E graph tests with mocked LLM and external tools.
Runs the full compiled graph (build_graph()) without real API calls.
Covers: CONSULT (missing params), PLAN (full pipeline), REVISE.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from graph.main_graph import build_graph
from state.schema import create_initial_state
from state.contracts import (
    Activity, Coordinates, Itinerary, ItineraryDay, QueryClassification,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_classification(intent: str, region: str = "Adjara", days: int | None = 2):
    return QueryClassification(
        intent=intent,
        region=region,
        search_query=f"Georgia {region} tourism",
        days=days,
        pace="moderate",
        should_ask_clarification=(intent == "CONSULT"),
        conversation_stage=intent,
        reasoning=f"test classification {intent}",
        steps=[],
        estimated_agents=1,
    )


def _make_itinerary(days: int = 2) -> Itinerary:
    return Itinerary(
        days=[
            ItineraryDay(
                day=i + 1,
                location="Batumi",
                activities=[
                    Activity(
                        name=f"Place{i * 2 + 1}",
                        location="Batumi",
                        category="attraction",
                        coordinates=Coordinates(lat=41.6, lon=41.6),
                        description="Test place",
                    ),
                    Activity(
                        name=f"Place{i * 2 + 2}",
                        location="Adjara",
                        category="nature",
                        coordinates=Coordinates(lat=41.7, lon=41.8),
                        description="Test nature",
                    ),
                ],
                total_distance_km=25.0,
                total_driving_hours=0.5,
            )
            for i in range(days)
        ],
        total_days=days,
        pace="moderate",
        summary=f"Test {days}-day Adjara trip",
    )


_FAKE_PLACES = [
    {
        "id": "p1", "name": "Batumi Boulevard", "location": "Batumi",
        "category": "attraction", "score": 0.9, "description": "Main promenade",
        "tags": ["walking"], "metadata": {"lat": 41.6168, "lon": 41.6367},
    },
    {
        "id": "p2", "name": "Mtirala National Park", "location": "Adjara",
        "category": "nature", "score": 0.85, "description": "Rainforest",
        "tags": ["hiking"], "metadata": {"lat": 41.72, "lon": 41.81},
    },
]

_FAKE_GEOCODE = {"lat": 41.6168, "lon": 41.6367, "display_name": "Batumi"}
_FAKE_ROUTE = {"distance": 2500, "duration": 1800}


@pytest.fixture
def graph():
    return build_graph(checkpointer=None)


@pytest.fixture
def base_state():
    return create_initial_state(
        user_query="Хочу поехать в Грузию",
        request_id="e2e-test-1",
        user_language="ru",
    )


@pytest.fixture
def plan_state():
    return create_initial_state(
        user_query="Батуми 2 дня природа",
        request_id="e2e-test-2",
        user_language="ru",
    )


# ---------------------------------------------------------------------------
# Test 1: CONSULT — missing params → questions returned, no itinerary
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_e2e_consult_missing_params(graph, base_state):
    """
    Classifier returns CONSULT with empty params → consultation_agent asks questions.
    Graph must NOT reach planning. task_complete=True, no itinerary days.
    """
    with patch(
        "orchestrator.orchestrator.OrchestratorAgent.classify",
        new_callable=AsyncMock,
        return_value=_make_classification("CONSULT", region="", days=None),
    ):
        config = {"configurable": {"thread_id": "e2e-consult-1"}}
        result = await graph.ainvoke(base_state, config=config)

    assert "consultation_agent" in result.get("agent_history", [])
    assert result.get("task_complete") is True
    assert result.get("final_response"), "Expected clarification questions in response"
    assert not result.get("raw_itinerary", {}).get("days"), \
        "CONSULT must not produce an itinerary"


# ---------------------------------------------------------------------------
# Test 2: PLAN — full pipeline search→geo→planning→validation→response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_e2e_plan_full_pipeline(graph, plan_state):
    """
    PLAN flow with all external calls mocked.
    planning_agent uses direct llm.ainvoke → patch langchain_openai.ChatOpenAI (lazy import).
    response_agent uses module-level import → patch agents.response.agent.ChatOpenAI.
    Verifies correct agent execution order (AT) and that each LLM was actually called (AU).
    """
    plan_itinerary = _make_itinerary(days=2)
    # planning_agent parses response.content as itinerary JSON
    plan_json = plan_itinerary.model_dump_json()

    # planning_agent: lazy import from langchain_openai
    mock_plan_response = MagicMock()
    mock_plan_response.content = plan_json
    mock_plan_llm = MagicMock()
    mock_plan_llm.ainvoke = AsyncMock(return_value=mock_plan_response)
    mock_planning_cls = MagicMock(return_value=mock_plan_llm)

    # response_agent: module-level import — must patch agents.response.agent.ChatOpenAI
    mock_resp_msg = MagicMock()
    mock_resp_msg.content = "Test Adjara travel response"
    mock_resp_llm = MagicMock()
    mock_resp_llm.ainvoke = AsyncMock(return_value=mock_resp_msg)
    mock_response_cls = MagicMock(return_value=mock_resp_llm)

    with (
        patch(
            "orchestrator.orchestrator.OrchestratorAgent.classify",
            new_callable=AsyncMock,
            return_value=_make_classification("PLAN", region="Adjara", days=2),
        ),
        patch("src.tools.tool_cache.cached_search_qdrant", new_callable=AsyncMock, return_value=_FAKE_PLACES),
        patch("src.tools.tool_cache.cached_geocode_city", new_callable=AsyncMock, return_value=_FAKE_GEOCODE),
        patch("src.tools.tool_cache.cached_get_route", new_callable=AsyncMock, return_value=_FAKE_ROUTE),
        patch("langchain_openai.ChatOpenAI", mock_planning_cls),
        patch("agents.response.agent.ChatOpenAI", mock_response_cls),
    ):
        config = {"configurable": {"thread_id": "e2e-plan-1"}}
        result = await graph.ainvoke(plan_state, config=config)

    # AT: agents must run in the correct pipeline order
    history = result.get("agent_history", [])
    for agent in ("search_agent", "geo_agent", "planning_agent", "validation_agent", "response_agent"):
        assert agent in history, f"Expected {agent} in agent_history, got {history}"
    assert history.index("search_agent") < history.index("geo_agent")
    assert history.index("geo_agent") < history.index("planning_agent")
    assert history.index("planning_agent") < history.index("validation_agent")
    assert history.index("validation_agent") < history.index("response_agent")

    # AU: each LLM was actually instantiated and invoked — routing bugs leave these uncalled
    # assumes ENABLE_EVAL=False; eval_node shares this patch and would make it called_twice
    mock_planning_cls.assert_called_once()         # planning_agent created ChatOpenAI
    mock_plan_llm.ainvoke.assert_called_once()     # planning_agent called ainvoke
    mock_response_cls.assert_called_once()         # response_agent created ChatOpenAI
    mock_resp_llm.ainvoke.assert_called_once()     # response_agent called ainvoke

    assert result.get("raw_itinerary", {}).get("days"), "Expected populated itinerary"
    assert result.get("final_response"), "Expected non-empty final response"


# ---------------------------------------------------------------------------
# Test 3: REVISE — revision→validation→response, no search/geo
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_e2e_revise_skips_search_and_geo(graph):
    """
    REVISE flow: existing 1-day plan revised to 2 days.
    revision_agent must run; search_agent and geo_agent must NOT run.
    raw_itinerary must reflect the revision (AV: day count changes from 1 to 2).
    """
    existing = _make_itinerary(days=1)

    state = create_initial_state(
        user_query="Добавь ещё один день",
        request_id="e2e-revise-1",
        user_language="ru",
    )
    state["has_current_plan"] = True
    state["current_plan"] = existing.model_dump()
    state["current_plan_version"] = 1

    # AV: revised plan is structurally different — 2 days instead of 1
    revised = _make_itinerary(days=2)

    # revision_agent: module-level import → patch agents.revision.agent.ChatOpenAI
    # uses prompt | with_structured_output chain; LangChain wraps the AsyncMock as RunnableLambda
    mock_rev_chain = AsyncMock(return_value=revised)
    mock_rev_llm = MagicMock()
    mock_rev_llm.with_structured_output = MagicMock(return_value=mock_rev_chain)

    # response_agent: module-level import → patch agents.response.agent.ChatOpenAI
    mock_resp_msg = MagicMock()
    mock_resp_msg.content = "Your revised trip plan is ready"
    mock_resp_llm = MagicMock()
    mock_resp_llm.ainvoke = AsyncMock(return_value=mock_resp_msg)

    with (
        patch(
            "orchestrator.orchestrator.OrchestratorAgent.classify",
            new_callable=AsyncMock,
            return_value=_make_classification("REVISE", region="Adjara", days=2),
        ),
        patch("agents.revision.agent.ChatOpenAI", return_value=mock_rev_llm),
        patch("agents.response.agent.ChatOpenAI", return_value=mock_resp_llm),
    ):
        config = {"configurable": {"thread_id": "e2e-revise-1"}}
        result = await graph.ainvoke(state, config=config)

    history = result.get("agent_history", [])
    assert "revision_agent" in history
    assert "validation_agent" in history
    assert "response_agent" in history
    assert "search_agent" not in history, f"REVISE must not trigger search, got {history}"
    assert "geo_agent" not in history, f"REVISE must not trigger geo, got {history}"

    # AV: revision must produce a structurally different itinerary (2 days, not 1)
    raw = result.get("raw_itinerary") or {}
    assert len(raw.get("days", [])) == 2, (
        f"Revision must change itinerary (expected 2 days, got {len(raw.get('days', []))})"
    )
    mock_rev_chain.assert_called_once()  # revision_agent actually invoked the LLM chain
