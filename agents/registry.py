"""Agent registry for the travel assistant graph."""

from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class AgentRegistry:
    def __init__(self):
        self._agents: dict[str, Callable] = {}
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        logger.info("AgentRegistry: initializing agent nodes...")
        from agents.search.agent import search_agent_node
        from agents.geo.agent import geo_agent_node
        from agents.planning.agent import planning_agent_node
        from agents.validation.agent import validation_agent_node
        from agents.response.agent import response_agent_node
        from agents.consultation.agent import consultation_agent_node
        from agents.revision.agent import revision_agent_node
        from agents.error_handler import safe_node

        self._agents = {
            "search_agent":       safe_node(search_agent_node,       "search_agent"),
            "geo_agent":          safe_node(geo_agent_node,          "geo_agent"),
            "planning_agent":     safe_node(planning_agent_node,     "planning_agent"),
            "validation_agent":   safe_node(validation_agent_node,   "validation_agent"),
            "response_agent":     safe_node(response_agent_node,     "response_agent"),
            "consultation_agent": safe_node(consultation_agent_node, "consultation_agent"),
            "revision_agent":     safe_node(revision_agent_node,     "revision_agent"),
        }
        self._initialized = True
        logger.info("AgentRegistry: %d agent nodes ready", len(self._agents))

    def get(self, name: str):
        if not self._initialized:
            self.initialize()
        return self._agents[name]


_registry: Optional[AgentRegistry] = None


def get_registry() -> AgentRegistry:
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry

