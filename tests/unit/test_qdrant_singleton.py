"""
Unit tests for QdrantClient singleton in search_tools.
"""
from __future__ import annotations

import threading
import pytest
from unittest.mock import patch, MagicMock


def _reset_singleton():
    import src.tools.search_tools as m
    m._qdrant_client = None


def test_get_qdrant_client_returns_singleton():
    """Two calls to get_qdrant_client() must return the same object."""
    _reset_singleton()

    fake_client = MagicMock()
    with patch("src.tools.search_tools.QdrantClient", return_value=fake_client) as mock_cls:
        from src.tools.search_tools import get_qdrant_client
        c1 = get_qdrant_client()
        c2 = get_qdrant_client()

    assert c1 is c2
    assert mock_cls.call_count == 1, "QdrantClient constructor must be called exactly once"


def test_get_qdrant_client_uses_settings():
    """Client is constructed with url/api_key from settings and timeout=10."""
    _reset_singleton()

    fake_client = MagicMock()
    with patch("src.tools.search_tools.QdrantClient", return_value=fake_client) as mock_cls:
        from src.tools.search_tools import get_qdrant_client
        get_qdrant_client()

    _, kwargs = mock_cls.call_args
    assert kwargs.get("timeout") == 10


@pytest.mark.asyncio
async def test_search_qdrant_reuses_singleton():
    """search_qdrant called twice uses the same QdrantClient instance."""
    _reset_singleton()

    fake_client = MagicMock()
    fake_client.query_points.return_value = MagicMock(points=[])

    with patch("src.tools.search_tools.QdrantClient", return_value=fake_client) as mock_cls, \
         patch("src.tools.embeddings.embed", return_value=[0.1] * 768):
        from src.tools.search_tools import search_qdrant
        import asyncio
        await asyncio.to_thread(search_qdrant.invoke, {"query": "churches", "top_k": 3})
        await asyncio.to_thread(search_qdrant.invoke, {"query": "mountains", "top_k": 3})

    assert mock_cls.call_count == 1, "QdrantClient must not be re-created on second search"


def test_get_qdrant_client_thread_safe():
    """Concurrent calls from multiple threads must produce exactly one QdrantClient."""
    _reset_singleton()

    call_counts = []
    clients = []

    def fake_constructor(**kwargs):
        call_counts.append(1)
        return MagicMock()

    def worker():
        from src.tools.search_tools import get_qdrant_client
        clients.append(get_qdrant_client())

    with patch("src.tools.search_tools.QdrantClient", side_effect=fake_constructor):
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert len(call_counts) == 1, f"QdrantClient constructed {len(call_counts)} times, expected 1"
    assert all(c is clients[0] for c in clients), "Not all threads got the same singleton"
