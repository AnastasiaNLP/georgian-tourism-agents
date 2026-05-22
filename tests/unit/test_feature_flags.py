# tests/unit/test_feature_flags.py
"""Tests for feature flags."""

import os
import pytest
from unittest.mock import patch


def _load_flags(env: dict) -> dict:
    with patch.dict(os.environ, env, clear=False):
        from graph.main_graph import _load_feature_flags
        return _load_feature_flags()


class TestFeatureFlags:
    def test_rag_false_disables(self):
        flags = _load_flags({"FEATURE_ENABLE_RAG": "false"})
        assert flags["ENABLE_RAG"] is False

    def test_rag_zero_disables(self):
        """FEATURE_ENABLE_RAG=0 disables RAG."""
        flags = _load_flags({"FEATURE_ENABLE_RAG": "0"})
        assert flags["ENABLE_RAG"] is False

    def test_rag_true_enables(self):
        flags = _load_flags({"FEATURE_ENABLE_RAG": "true"})
        assert flags["ENABLE_RAG"] is True

    def test_geo_routes_false_disables(self):
        flags = _load_flags({"FEATURE_ENABLE_GEO_ROUTES": "false"})
        assert flags["ENABLE_GEO_ROUTES"] is False

    def test_geo_routes_zero_disables(self):
        """FEATURE_ENABLE_GEO_ROUTES=0 disables geo routes."""
        flags = _load_flags({"FEATURE_ENABLE_GEO_ROUTES": "0"})
        assert flags["ENABLE_GEO_ROUTES"] is False

    def test_validation_flag(self):
        flags = _load_flags({"FEATURE_ENABLE_VALIDATION": "false"})
        assert flags["ENABLE_VALIDATION"] is False

    def test_eval_default_false(self):
        flags = _load_flags({})
        assert flags["ENABLE_EVAL"] is False

    def test_weather_default_false(self):
        flags = _load_flags({})
        assert flags["USE_WEATHER"] is False
