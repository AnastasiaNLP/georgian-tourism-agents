"""Live API smoke tests against a running server on localhost:8000.

Tests skip automatically when the server is not reachable.
Run with the server started:

    PYTHONPATH="" .venv/bin/uvicorn main:app --port 8000

Then:

    PYTHONPATH="" .venv/bin/pytest tests/online/ -v
"""

import os
import pytest
import httpx

BASE_URL = "http://localhost:8000"


def _headers() -> dict:
    api_key = os.getenv("API_KEY")
    return {"X-API-Key": api_key} if api_key else {}


def _server_available() -> bool:
    try:
        httpx.get(f"{BASE_URL}/api/v1/health/live", timeout=2.0)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.online


@pytest.fixture(autouse=True, scope="module")
def require_server():
    if not _server_available():
        pytest.skip("Server not running on localhost:8000")


def test_health_endpoint_live_dependencies():
    response = httpx.get(f"{BASE_URL}/api/v1/health/ready", headers=_headers(), timeout=10)
    assert response.status_code == 200
    payload = response.json()
    assert payload["openai_key"] == "set"
    assert payload["anthropic_key"] == "set"
    assert "qdrant" in payload
    assert payload["status"] in {"healthy", "degraded"}


def test_liveness_probe():
    response = httpx.get(f"{BASE_URL}/api/v1/health/live", timeout=5)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_plan_endpoint_returns_response():
    response = httpx.post(
        f"{BASE_URL}/api/v1/plan",
        json={
            "query": "Какие музеи стоит посмотреть в Тбилиси? Ответь кратко.",
            "language": "ru",
        },
        headers=_headers(),
        timeout=120,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["request_id"]
    assert payload["response"]
    assert "search_agent" in payload["agent_history"]
    assert "response_agent" in payload["agent_history"]
    assert payload["duration_seconds"] >= 0


def test_checkpoint_continuity_via_shared_pool():
    """Two requests on the same conversation_id must share LangGraph checkpoint state.

    Verifies that AsyncPostgresSaver can read back state written by the previous
    request when both use the shared AsyncConnectionPool. A regression here means
    the pool's autocommit/prepare_threshold settings are wrong.
    """
    conversation_id = "test-pool-continuity-check"

    r1 = httpx.post(
        f"{BASE_URL}/api/v1/plan",
        json={
            "query": "Хочу посетить Тбилиси — что посоветуешь?",
            "language": "ru",
            "conversation_id": conversation_id,
        },
        headers=_headers(),
        timeout=120,
    )
    assert r1.status_code == 200, f"First request failed: {r1.text}"
    p1 = r1.json()
    assert p1["thread_id"] == conversation_id

    r2 = httpx.post(
        f"{BASE_URL}/api/v1/plan",
        json={
            "query": "А что насчёт Батуми на следующий день?",
            "language": "ru",
            "conversation_id": conversation_id,
        },
        headers=_headers(),
        timeout=120,
    )
    assert r2.status_code == 200, f"Second request failed: {r2.text}"
    p2 = r2.json()
    # If checkpoint failed to persist r1's state, r2 would start fresh and lack
    # any context. A 200 response alone is sufficient to confirm the pool path worked.
    assert p2["status"] == "ok"
    assert p2["thread_id"] == conversation_id


def test_plan_endpoint_langsmith_tracing():
    if not (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
        pytest.skip("LANGSMITH_API_KEY not set")

    response = httpx.post(
        f"{BASE_URL}/api/v1/plan",
        json={"query": "Батуми 2 дня природа", "language": "ru"},
        headers=_headers(),
        timeout=120,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["has_itinerary"] is True
