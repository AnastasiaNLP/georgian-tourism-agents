# tests/unit/test_error_handler.py
"""Tests for error cascade logic."""

import pytest
from agents.error_handler import _escalate, TRANSIENT_ERRORS


class TestEscalate:
    def test_transient_in_normal_stays_normal(self):
        """A timeout in normal mode does not move the system to degraded."""
        state = {"execution_mode": "normal"}
        result = _escalate(state, "geo_agent", "timeout 60s", error_type="TimeoutError")
        assert result["execution_mode"] == "normal"

    def test_permanent_in_normal_goes_degraded(self):
        state = {"execution_mode": "normal"}
        result = _escalate(state, "geo_agent", "ValueError", error_type="ValueError")
        assert result["execution_mode"] == "degraded"

    def test_transient_in_degraded_goes_emergency(self):
        """A repeated transient error moves the system to emergency."""
        state = {"execution_mode": "degraded"}
        result = _escalate(state, "geo_agent", "timeout again", error_type="TimeoutError")
        assert result["execution_mode"] == "emergency"

    def test_permanent_in_degraded_goes_emergency(self):
        state = {"execution_mode": "degraded"}
        result = _escalate(state, "planning_agent", "crash", error_type="RuntimeError")
        assert result["execution_mode"] == "emergency"

    def test_error_added_to_list(self):
        state = {"execution_mode": "normal"}
        result = _escalate(state, "search_agent", "oops", error_type="ValueError")
        assert any("search_agent" in e for e in result["errors"])

    def test_agent_history_added(self):
        state = {"execution_mode": "normal"}
        result = _escalate(state, "geo_agent", "oops", error_type="ValueError")
        assert "geo_agent" in result["agent_history"]

    def test_connection_error_is_transient(self):
        state = {"execution_mode": "normal"}
        result = _escalate(state, "geo_agent", "conn reset", error_type="ConnectionError")
        assert result["execution_mode"] == "normal"

    def test_transient_errors_set_covers_key_types(self):
        assert "TimeoutError" in TRANSIENT_ERRORS
        assert "ConnectionError" in TRANSIENT_ERRORS
        assert "RateLimitError" in TRANSIENT_ERRORS
