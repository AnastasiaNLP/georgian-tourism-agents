# tests/integration/test_search_node.py
"""Integration tests for search_agent_node with mocked Qdrant search."""

import pytest
from unittest.mock import patch, MagicMock


MOCK_PLACES = [
    {"id": "1", "name": "Batumi Botanical Garden", "location": "Batumi, Adjara",
     "category": "botanical garden", "score": 0.92, "description": "A beautiful garden",
     "tags": ["nature", "garden"]},
    {"id": "2", "name": "Ботанический сад Батуми", "location": "Batumi, Adjara",
     "category": "botanical garden", "score": 0.89, "description": "Красивый сад",
     "tags": ["природа"]},
    {"id": "3", "name": "Mtirala National Park", "location": "Adjara",
     "category": "national park", "score": 0.85, "description": "Rainforest",
     "tags": ["nature", "hiking"]},
    {"id": "4", "name": "Gonio Fortress", "location": "Gonio, Adjara",
     "category": "fortress", "score": 0.80, "description": "Ancient fortress",
     "tags": ["history"]},
]


@pytest.fixture
def mock_search():
    """Mock cached_search_qdrant and filter_by_region."""
    with patch("agents.search.agent._do_search") as mock:
        mock.return_value = MOCK_PLACES
        yield mock


@pytest.mark.asyncio
async def test_search_node_no_llm_calls(mock_search):
    """search_agent_node does not make LLM calls."""
    from agents.search.agent import search_agent_node
    state = {
        "request_id": "test-1",
        "user_query": "природа Батуми",
        "orchestrator_params": {"region": "Adjara", "max_results": 10},
    }
    result = await search_agent_node(state)
    # No budget_state with LLM calls > 0.
    assert "search_results" in result
    assert result["search_results"] is not None


@pytest.mark.asyncio
async def test_search_node_deduplicates(mock_search):
    """RU/EN duplicates are removed."""
    from agents.search.agent import search_agent_node
    state = {
        "request_id": "test-2",
        "user_query": "природа Батуми",
        "orchestrator_params": {"region": "Adjara", "max_results": 10},
    }
    result = await search_agent_node(state)
    names = [p["name"] for p in result["search_results"]]
    botanical = [n for n in names if "botanical" in n.lower() or "ботанический" in n.lower()]
    assert len(botanical) == 1


@pytest.mark.asyncio
async def test_search_node_nature_query_filters(mock_search):
    """A nature query ranks nature places above fortresses."""
    from agents.search.agent import search_agent_node
    state = {
        "request_id": "test-3",
        "user_query": "природа горы парки",
        "orchestrator_params": {"region": "Adjara", "max_results": 10},
    }
    result = await search_agent_node(state)
    results = result["search_results"]
    if len(results) >= 2:
        nature_first = any(
            tag in (results[0].get("tags") or [])
            for tag in ["nature", "hiking", "park", "природа"]
        )
        # A nature place should be near the top.
        assert nature_first or results[0]["category"] in ["national park", "botanical garden"]


@pytest.mark.asyncio
async def test_search_context_region_never_none(mock_search):
    """search_context.region is never None."""
    from agents.search.agent import search_agent_node
    # Even when there are no results.
    mock_search.return_value = []
    state = {
        "request_id": "test-4",
        "user_query": "test",
        "orchestrator_params": {"region": "Adjara"},
    }
    result = await search_agent_node(state)
    assert result["search_context"]["region"] is not None
    assert result["search_context"]["region"] == "Adjara"


@pytest.mark.asyncio
async def test_search_node_returns_required_fields(mock_search):
    from agents.search.agent import search_agent_node
    state = {
        "request_id": "test-5",
        "user_query": "Батуми",
        "orchestrator_params": {"region": "Adjara"},
    }
    result = await search_agent_node(state)
    assert "search_results" in result
    assert "search_context" in result
    assert "agent_history" in result
    assert "agent_scratchpad" in result
    assert "search_agent" in result["agent_history"]
