# tests/unit/test_fallback_rules.py
"""Unit tests for orchestrator/fallback_rules.py."""

import pytest
from state.models import OrchestratorPlan, OrchestratorStep
from orchestrator.fallback_rules import (
    enforce_plan_rules,
    create_fallback_plan,
    create_fallback_review,
)


def _make_plan(*agents) -> OrchestratorPlan:
    steps = [OrchestratorStep(agent=a, params={}, reason="test") for a in agents]
    return OrchestratorPlan(steps=steps, reasoning="test", estimated_agents=len(steps))


class TestEnforcePlanRules:
    def test_geo_inserted_before_planning(self):
        plan = _make_plan("search_agent", "planning_agent", "response_agent")
        result = enforce_plan_rules(plan, {})
        agents = [s.agent for s in result.steps]
        geo_idx = agents.index("geo_agent")
        plan_idx = agents.index("planning_agent")
        assert geo_idx < plan_idx

    def test_geo_moved_before_planning_when_after(self):
        """geo after planning should be moved before planning."""
        plan = _make_plan("search_agent", "planning_agent", "geo_agent", "response_agent")
        result = enforce_plan_rules(plan, {})
        agents = [s.agent for s in result.steps]
        assert agents.index("geo_agent") < agents.index("planning_agent")

    def test_search_always_first(self):
        plan = _make_plan("planning_agent", "response_agent")
        result = enforce_plan_rules(plan, {"user_query": "Батуми"})
        assert result.steps[0].agent == "search_agent"

    def test_response_always_last(self):
        plan = _make_plan("search_agent", "planning_agent")
        result = enforce_plan_rules(plan, {})
        assert result.steps[-1].agent == "response_agent"

    def test_validation_added_when_planning(self):
        plan = _make_plan("search_agent", "planning_agent", "response_agent")
        result = enforce_plan_rules(plan, {})
        agents = [s.agent for s in result.steps]
        assert "validation_agent" in agents

    def test_validation_before_response(self):
        plan = _make_plan("search_agent", "planning_agent", "response_agent")
        result = enforce_plan_rules(plan, {})
        agents = [s.agent for s in result.steps]
        assert agents.index("validation_agent") < agents.index("response_agent")

    def test_invalid_agents_removed(self):
        steps = [
            OrchestratorStep(agent="search_agent", params={}, reason=""),
            OrchestratorStep(agent="unknown_agent", params={}, reason=""),
            OrchestratorStep(agent="response_agent", params={}, reason=""),
        ]
        plan = OrchestratorPlan(steps=steps, reasoning="", estimated_agents=3)
        result = enforce_plan_rules(plan, {})
        agents = [s.agent for s in result.steps]
        assert "unknown_agent" not in agents

    def test_max_7_steps(self):
        plan = _make_plan(*["search_agent"] + ["geo_agent"] * 10 + ["response_agent"])
        result = enforce_plan_rules(plan, {})
        assert len(result.steps) <= 7

    def test_no_consecutive_duplicates(self):
        steps = [
            OrchestratorStep(agent="search_agent", params={}, reason=""),
            OrchestratorStep(agent="search_agent", params={}, reason=""),
            OrchestratorStep(agent="response_agent", params={}, reason=""),
        ]
        plan = OrchestratorPlan(steps=steps, reasoning="", estimated_agents=3)
        result = enforce_plan_rules(plan, {})
        agents = [s.agent for s in result.steps]
        for i in range(len(agents) - 1):
            assert agents[i] != agents[i + 1]


def test_detect_region_no_hardcoded_keywords():
    """_detect_region must read from config/georgia_regions.json, not from a hardcoded dict."""
    import inspect
    from orchestrator.fallback_rules import _detect_region

    src = inspect.getsource(_detect_region)
    assert "region_keywords" not in src, (
        "_detect_region must not use a hardcoded region_keywords dict (golden rule: no hardcoded regions)"
    )
    assert "load_regions" in src, "_detect_region must delegate to load_regions()"


class TestCreateFallbackPlan:
    def test_planning_query_includes_geo(self):
        state = {"user_query": "Батуми 3 дня природа"}
        result = create_fallback_plan(state)
        agents = [s.agent for s in result.steps]
        assert "geo_agent" in agents

    def test_geo_before_planning_in_fallback(self):
        state = {"user_query": "plan 3 days Batumi"}
        result = create_fallback_plan(state)
        agents = [s.agent for s in result.steps]
        assert agents.index("geo_agent") < agents.index("planning_agent")

    def test_simple_query_no_planning(self):
        state = {"user_query": "что посмотреть в Тбилиси"}
        result = create_fallback_plan(state)
        agents = [s.agent for s in result.steps]
        assert "planning_agent" not in agents
        assert "search_agent" in agents
        assert "response_agent" in agents

    def test_region_detected_batumi(self):
        state = {"user_query": "поездка в Батуми"}
        result = create_fallback_plan(state)
        search_step = next(s for s in result.steps if s.agent == "search_agent")
        assert search_step.params.get("region") == "Adjara"

    def test_always_ends_with_response(self):
        state = {"user_query": "3 дня в Грузии"}
        result = create_fallback_plan(state)
        assert result.steps[-1].agent == "response_agent"


class TestCreateFallbackReview:
    def test_after_search_calls_geo(self):
        state = {
            "agent_history": ["search_agent"],
            "search_results": [{"name": "Place"}],
        }
        result = create_fallback_review(state)
        assert result.agent == "geo_agent"

    def test_after_geo_calls_planning(self):
        state = {
            "agent_history": ["search_agent", "geo_agent"],
            "search_results": [{"name": "Place"}],
            "enriched_places": [{"name": "Place", "lat": 41.6, "lon": 41.6}],
        }
        result = create_fallback_review(state)
        assert result.agent == "planning_agent"

    def test_after_planning_calls_validation(self):
        state = {
            "agent_history": ["search_agent", "geo_agent", "planning_agent"],
        }
        result = create_fallback_review(state)
        assert result.agent == "validation_agent"

    def test_after_validation_done(self):
        state = {
            "agent_history": ["search_agent", "geo_agent", "planning_agent", "validation_agent"],
        }
        result = create_fallback_review(state)
        assert result.next_action == "done"

    def test_after_response_done(self):
        state = {"agent_history": ["search_agent", "response_agent"]}
        result = create_fallback_review(state)
        assert result.next_action == "done"
