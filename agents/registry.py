# agents/registry.py
"""
AgentRegistry — singleton registry for graph agent nodes.
"""
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class AgentRegistry:
    _instance: Optional["AgentRegistry"] = None

    def __new__(cls) -> "AgentRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._nodes = {}
            cls._instance._initialized = False
        return cls._instance

    def initialize(self) -> None:
        """Initialize all agent nodes once at startup."""
        if self._initialized:
            return

        logger.info("AgentRegistry: initializing agent nodes...")

        from agents.geo.agent import geo_agent_node
        from agents.search.agent import search_agent_node
        from agents.planning.agent import planning_agent_node
        from agents.validation.agent import validation_agent_node
        from agents.response.agent import response_agent_node
        from agents.error_handler import safe_node
        from monitoring.metrics import track_agent

        self._nodes = {
            "search_agent":     track_agent("search_agent")(safe_node(search_agent_node,     "search_agent")),
            "planning_agent":   track_agent("planning_agent")(safe_node(planning_agent_node,   "planning_agent")),
            "geo_agent":        track_agent("geo_agent")(safe_node(geo_agent_node,        "geo_agent")),
            "validation_agent": track_agent("validation_agent")(safe_node(validation_agent_node, "validation_agent")),
            "response_agent":   track_agent("response_agent")(safe_node(response_agent_node,   "response_agent")),
        }

        self._initialized = True
        logger.info(f"AgentRegistry: {len(self._nodes)} agent nodes ready")

    def get(self, name: str) -> Callable:
        if not self._initialized:
            self.initialize()
        if name not in self._nodes:
            raise KeyError(f"Agent '{name}' not found. Available: {list(self._nodes.keys())}")
        return self._nodes[name]


_registry = AgentRegistry()

def get_registry() -> AgentRegistry:
    return _registry
