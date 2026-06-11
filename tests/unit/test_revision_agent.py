from __future__ import annotations

import json

import pytest

from agents.revision import agent as revision_agent_module
from agents.revision.agent import revision_agent_node

class _FakeRevisionResult:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def model_dump(self) -> dict[str, object]:
        return self._payload


class _FakeChain:
    def __init__(self, result: _FakeRevisionResult) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []
        self.configs: list[dict[str, object]] = []

    async def ainvoke(self, inputs: dict[str, object], config: dict[str, object] | None = None):
        self.calls.append(inputs)
        self.configs.append(config or {})
        return self.result


class _FakePrompt:
    def __init__(self, chain: _FakeChain) -> None:
        self.chain = chain

    def __or__(self, _other):
        return self.chain


class _FakeLLM:
    def with_structured_output(self, _schema):
        return object()


class _FakeTracker:
    def __init__(self, *args, **kwargs) -> None:
        self.model = kwargs.get("model", "")


@pytest.mark.asyncio
async def test_revision_agent_updates_raw_itinerary_and_version(monkeypatch):
    revised_plan = {
        "days": [
            {
                "day": 1,
                "location": "Tbilisi",
                "activities": [{"name": "Old Tbilisi walk"}],
                "total_distance_km": 5.0,
                "total_driving_hours": 0.0,
            }
        ],
        "total_days": 1,
        "pace": "moderate",
        "summary": "Revised plan",
    }
    fake_chain = _FakeChain(_FakeRevisionResult(revised_plan))

    monkeypatch.setattr(revision_agent_module, "get_revision_prompt", lambda: _FakePrompt(fake_chain))
    monkeypatch.setattr(revision_agent_module, "ChatOpenAI", lambda *args, **kwargs: _FakeLLM())
    monkeypatch.setattr(revision_agent_module, "TokenTracker", _FakeTracker)
    monkeypatch.setattr(
        revision_agent_module,
        "merge_budget",
        lambda budget_state, tracker: {**budget_state, "tracked": True},
    )

    state = {
        "request_id": "req-1",
        "current_plan_version": 0,
        "current_plan": {
            "days": [
                {
                    "day": 1,
                    "location": "Kutaisi",
                    "activities": [{"name": "Previous plan"}],
                    "total_distance_km": 10.0,
                    "total_driving_hours": 1.0,
                }
            ],
            "total_days": 1,
            "pace": "moderate",
        },
        "raw_itinerary": {},
        "feedback": "убери третий день",
        "trip_parameters": {"region": "Georgia", "days": 1, "pace": "moderate", "search_query": "Georgia"},
        "budget_state": {"tokens": 10},
        "agent_history": [],
        "agent_scratchpad": {},
        "router_trace": [],
        "errors": [],
        "has_current_plan": True,
    }

    result = await revision_agent_node(state)

    assert result["current_plan_version"] == 1
    assert result["current_plan"] == revised_plan
    assert result["raw_itinerary"] == revised_plan
    assert result["enriched_itinerary"] == {}
    assert result["conversation_stage"] == "REVISE"
    assert result["budget_state"] == {"tokens": 10, "tracked": True}
    assert fake_chain.calls

    payload = fake_chain.calls[0]
    assert json.loads(payload["current_plan"]) == state["current_plan"]
    assert json.loads(payload["trip_parameters"]) == state["trip_parameters"]
    assert payload["feedback"] == state["feedback"]


@pytest.mark.asyncio
async def test_revision_agent_no_plan_routes_to_validation(monkeypatch):
    monkeypatch.setattr(revision_agent_module, "get_revision_prompt", lambda: _FakePrompt(_FakeChain(_FakeRevisionResult({}))))
    monkeypatch.setattr(revision_agent_module, "ChatOpenAI", lambda *args, **kwargs: _FakeLLM())
    monkeypatch.setattr(revision_agent_module, "TokenTracker", _FakeTracker)
    monkeypatch.setattr(
        revision_agent_module,
        "merge_budget",
        lambda budget_state, tracker: {**budget_state, "tracked": True},
    )

    result = await revision_agent_node(
        {
            "request_id": "req-2",
            "current_plan": {},
            "raw_itinerary": {},
            "feedback": "change something",
            "trip_parameters": {},
            "budget_state": {},
            "agent_history": [],
            "agent_scratchpad": {},
            "router_trace": [],
            "errors": [],
            "has_current_plan": False,
        }
    )

    assert result["router_trace"][-1]["next"] == "validation_agent"
    assert result["errors"] == ["revision_agent: no plan to revise"]
    assert result["raw_itinerary"] == {}


@pytest.mark.asyncio
async def test_revision_agent_chatOpenAI_receives_api_key(monkeypatch):
    """ChatOpenAI must be constructed with api_key from settings."""
    captured: dict = {}

    def _capturing_llm(**kwargs):
        captured.update(kwargs)
        return _FakeLLM()

    fake_chain = _FakeChain(_FakeRevisionResult({
        "days": [{"day": 1, "location": "Tbilisi", "activities": [{"name": "walk"}],
                  "total_distance_km": 0.0, "total_driving_hours": 0.0}],
        "total_days": 1, "pace": "moderate", "summary": "ok",
    }))

    monkeypatch.setattr(revision_agent_module, "get_revision_prompt", lambda: _FakePrompt(fake_chain))
    monkeypatch.setattr(revision_agent_module, "ChatOpenAI", _capturing_llm)
    monkeypatch.setattr(revision_agent_module, "TokenTracker", _FakeTracker)
    monkeypatch.setattr(revision_agent_module, "merge_budget",
                        lambda budget_state, tracker: budget_state)

    await revision_agent_node({
        "request_id": "req-ak",
        "current_plan_version": 0,
        "current_plan": {"days": [{"day": 1, "location": "X", "activities": [],
                                   "total_distance_km": 0.0, "total_driving_hours": 0.0}],
                         "total_days": 1, "pace": "moderate"},
        "raw_itinerary": {},
        "feedback": "change it",
        "trip_parameters": {"region": "Georgia", "days": 1, "pace": "moderate", "search_query": "q"},
        "budget_state": {},
        "agent_history": [],
        "agent_scratchpad": {},
        "router_trace": [],
        "errors": [],
        "has_current_plan": True,
    })

    assert "api_key" in captured, "ChatOpenAI must receive api_key kwarg"
    assert captured["api_key"] is not None

