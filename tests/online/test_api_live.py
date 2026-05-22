"""Live API smoke tests.

These tests call external services and may spend provider credits.
Run explicitly:

    RUN_ONLINE_TESTS=true pytest -m online tests/online -q

The test process must receive environment variables from the shell. The
application does not load `.env`.
"""

import os

import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.online


def _require_online(*names: str) -> None:
    if os.getenv("RUN_ONLINE_TESTS", "").lower() != "true":
        pytest.skip("Set RUN_ONLINE_TESTS=true to run live API tests")
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        pytest.skip(f"Missing live API env vars: {', '.join(missing)}")


def _headers() -> dict:
    api_key = os.getenv("API_KEY")
    return {"X-API-Key": api_key} if api_key else {}


def test_health_endpoint_live_dependencies():
    _require_online(
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "QDRANT_URL",
        "QDRANT_API_KEY",
    )

    from main import app

    client = TestClient(app)
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["openai_key"] == "set"
    assert payload["anthropic_key"] == "set"
    assert "qdrant" in payload
    assert payload["status"] in {"healthy", "degraded"}


def test_plan_endpoint_live_with_langsmith_tracing():
    _require_online(
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "QDRANT_URL",
        "QDRANT_API_KEY",
    )
    if not (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
        pytest.skip("Missing LANGSMITH_API_KEY or LANGCHAIN_API_KEY for tracing")

    from monitoring.tracing import setup_langsmith_tracing
    from main import app

    assert setup_langsmith_tracing() is True
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGCHAIN_API_KEY"]

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/plan",
            json={
                "query": "Какие музеи стоит посмотреть в Тбилиси? Ответь кратко.",
                "language": "ru",
            },
            headers=_headers(),
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["request_id"]
    assert payload["response"]
    assert "search_agent" in payload["agent_history"]
    assert "response_agent" in payload["agent_history"]
    assert payload["duration_seconds"] >= 0
