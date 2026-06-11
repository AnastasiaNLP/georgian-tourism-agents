"""Unit tests for stage-based routing."""


def test_route_from_orchestrator_primary_intent():
    """Primary path: orchestration.intent is the routing signal (not conversation_stage)."""
    from graph.main_graph import route_from_orchestrator

    assert route_from_orchestrator({"orchestration": {"intent": "CONSULT"}}) == "consultation_agent"
    assert route_from_orchestrator({"orchestration": {"intent": "REVISE"}}) == "revision_agent"
    assert route_from_orchestrator({"orchestration": {"intent": "PLAN"}}) == "search_agent"
    assert route_from_orchestrator({"orchestration": {"intent": "INFO"}}) == "search_agent"
    assert route_from_orchestrator({"orchestration": {"intent": "SEARCH"}}) == "search_agent"


def test_route_from_orchestrator_fallback_stage():
    """Fallback path: when orchestration.intent is absent, falls back to conversation_stage."""
    from graph.main_graph import route_from_orchestrator

    assert route_from_orchestrator({"conversation_stage": "CONSULT"}) == "consultation_agent"
    assert route_from_orchestrator({"conversation_stage": "REVISE"}) == "revision_agent"
    assert route_from_orchestrator({"conversation_stage": "PLAN"}) == "search_agent"
    assert route_from_orchestrator({"conversation_stage": "INFO"}) == "search_agent"


def test_route_after_search():
    from graph.main_graph import route_after_search

    assert route_after_search({"orchestration": {"intent": "PLAN"}}) == "geo_agent"
    assert route_after_search({"orchestration": {"intent": "INFO"}}) == "response_agent"
    assert route_after_search({"orchestration": {"intent": "SEARCH"}}) == "response_agent"
    # conversation_stage fallback when orchestration absent
    assert route_after_search({"conversation_stage": "PLAN"}) == "geo_agent"
    assert route_after_search({"conversation_stage": "INFO"}) == "response_agent"


def test_route_after_consultation_can_continue_to_search():
    from graph.main_graph import route_after_consultation

    assert route_after_consultation({"orchestration": {"intent": "PLAN"}}) == "search_agent"
    assert route_after_consultation({"orchestration": {"intent": "CONSULT"}}) == "response_agent"


def test_route_after_consultation_finalizes_questions():
    from graph.main_graph import route_after_consultation

    assert route_after_consultation({"task_complete": True}) == "memory_save"
    assert route_after_consultation({"task_complete": True, "orchestration": {"intent": "CONSULT"}}) == "memory_save"
