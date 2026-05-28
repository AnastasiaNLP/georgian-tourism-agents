"""Tests for deterministic graph routing."""


def test_search_intent_goes_to_response():
    from graph.main_graph import route_after_search

    assert route_after_search({"intent": "SEARCH"}) == "response_agent"


def test_plan_intent_goes_to_geo():
    from graph.main_graph import route_after_search

    assert route_after_search({"intent": "PLAN"}) == "geo_agent"


def test_validation_flag_controls_post_planning_route():
    from graph.main_graph import route_after_planning

    assert route_after_planning({"feature_flags": {"ENABLE_VALIDATION": True}}) == "validation_agent"
    assert route_after_planning({"feature_flags": {"ENABLE_VALIDATION": False}}) == "response_agent"


def test_graph_compiles():
    from graph.main_graph import build_graph

    graph = build_graph()
    assert type(graph).__name__ == "CompiledStateGraph"


def test_orchestrator_params_are_merged_from_steps():
    from graph.main_graph import _merged_orchestrator_params

    steps = [
        {
            "agent": "search_agent",
            "params": {
                "region": "Kakheti",
                "search_query": "wine vineyard monastery",
                "max_results": 10,
            },
        },
        {
            "agent": "planning_agent",
            "params": {"days": 2, "pace": "relaxed"},
        },
    ]

    assert _merged_orchestrator_params(steps) == {
        "region": "Kakheti",
        "search_query": "wine vineyard monastery",
        "max_results": 10,
        "days": 2,
        "pace": "relaxed",
    }
