# tests/integration/test_orchestrator_node.py
"""orchestrator_plan_node: conversation_stage stays consistent with the final intent."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from state.contracts import QueryClassification


def _classification(**kwargs) -> QueryClassification:
    base = dict(intent="PLAN", region="", search_query="trip",
                days=None, pace="moderate", conversation_stage="PLAN")
    base.update(kwargs)
    return QueryClassification(**base)


@pytest.mark.asyncio
async def test_consent_guard_syncs_stage_to_intent():
    """PLAN without days/region is downgraded to CONSULT — stage must follow, not stay PLAN."""
    from graph import main_graph

    fake_orch = MagicMock()
    fake_orch.classify = AsyncMock(return_value=_classification(days=None, region=""))

    with patch.object(main_graph, "_get_orchestrator", return_value=fake_orch):
        result = await main_graph.orchestrator_plan_node(
            {"request_id": "t", "user_query": "plan me a trip"}
        )

    assert result["intent"] == "CONSULT"
    assert result["conversation_stage"] == "CONSULT"
    assert result["orchestration"]["intent"] == "CONSULT"


@pytest.mark.asyncio
async def test_plan_with_params_keeps_stage_equal_to_intent():
    """A complete PLAN request keeps stage == intent == PLAN."""
    from graph import main_graph

    fake_orch = MagicMock()
    fake_orch.classify = AsyncMock(
        return_value=_classification(days=3, region="Kakheti")
    )

    with patch.object(main_graph, "_get_orchestrator", return_value=fake_orch):
        result = await main_graph.orchestrator_plan_node(
            {"request_id": "t", "user_query": "3 days in Kakheti"}
        )

    assert result["intent"] == "PLAN"
    assert result["conversation_stage"] == "PLAN"
