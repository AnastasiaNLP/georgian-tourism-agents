"""Unit tests for thread_id derivation in the API router."""


def test_derive_thread_id_uses_conversation_id_when_provided():
    from api.router import _derive_thread_id

    result = _derive_thread_id("user-123", "conv-abc")
    assert result == "conv-abc"


def test_derive_thread_id_stable_for_same_user():
    from api.router import _derive_thread_id

    a = _derive_thread_id("alice@example.com", None)
    b = _derive_thread_id("alice@example.com", None)
    assert a == b
    assert a.startswith("user-")


def test_derive_thread_id_different_users_different_threads():
    from api.router import _derive_thread_id

    a = _derive_thread_id("alice", None)
    b = _derive_thread_id("bob", None)
    assert a != b


def test_derive_thread_id_anonymous_unique():
    from api.router import _derive_thread_id

    a = _derive_thread_id(None, None)
    b = _derive_thread_id(None, None)
    assert a != b
    assert a.startswith("anon-")


def test_get_run_config_uses_thread_id_not_request_id():
    from monitoring.tracing import get_run_config

    config = get_run_config(
        request_id="req-999",
        thread_id="user-abc123",
        user_id="alice",
    )
    assert config["configurable"]["thread_id"] == "user-abc123"
    assert config["metadata"]["request_id"] == "req-999"
    assert config["metadata"]["user_id"] == "alice"
