"""
Integration test: two simultaneous requests for the same conversation_id must
produce exactly one 200 and one 409 (thread_busy).

Requires a live server. Run with:
    pytest tests/integration/test_race_condition.py -m online -v
"""
import asyncio
import pytest
import httpx

BASE_URL = "http://localhost:8000/api/v1"
HEADERS = {"X-API-Key": "test-key"}


@pytest.mark.online
@pytest.mark.asyncio
async def test_concurrent_same_thread_returns_409():
    payload = {
        "query": "Посоветуй тур по Грузии на 5 дней",
        "conversation_id": "race-test-conv-001",
        "language": "ru",
    }

    async def post():
        async with httpx.AsyncClient(timeout=120) as client:
            return await client.post(f"{BASE_URL}/plan", json=payload, headers=HEADERS)

    tasks = [asyncio.create_task(post()), asyncio.create_task(post())]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED, timeout=15)

    for t in pending:
        t.cancel()

    assert done, "No response received within 15 seconds — server may be down"

    result = done.pop().result()
    assert result.status_code == 409, (
        f"Expected 409 from the faster response, got {result.status_code}"
    )
    body = result.json()
    assert body.get("detail", {}).get("error") == "thread_busy"
