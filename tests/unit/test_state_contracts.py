"""Tests for canonical state contracts."""


def test_to_canonical_state_uses_defaults_for_none():
    from state.schema import to_canonical_state

    result = to_canonical_state(
        {
            "request_id": "r1",
            "user_query": "Batumi 2 days",
            "user_language": "en",
            "trip_parameters": None,
            "user_memory": None,
            "validation_result": None,
            "orchestration": None,
            "agent_history": [],
            "errors": [],
            "router_trace": [],
        }
    )
    assert result.request_id == "r1"
    assert result.trip_parameters.region == ""
    assert result.conversation_stage == "CONSULT"
    assert result.has_current_plan is False
    assert result.current_plan_version == 0
