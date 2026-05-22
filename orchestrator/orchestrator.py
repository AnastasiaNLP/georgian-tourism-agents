# orchestrator/orchestrator.py
"""Claude-based request classifier for deterministic graph routing."""

import logging
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from state.models import OrchestratorPlan
from orchestrator.prompts import load_prompt
from orchestrator.fallback_rules import (
    create_fallback_plan,
    enforce_plan_rules,
)

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Claude Haiku based request classifier.

    Runtime graph uses Claude once at the beginning to extract intent and
    parameters. Agent routing after that is deterministic in graph/main_graph.py.
    """

    def __init__(self):
        from config.settings import get_settings

        settings = get_settings()

        # Claude Haiku is fast and cost-efficient for request classification.
        self.llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            api_key=settings.anthropic_api_key,
            max_tokens=1000,
            temperature=0.0,
        )
        # Load the system prompt from YAML.
        self.system_prompt = load_prompt("orchestrator")
        logger.info("OrchestratorAgent classifier initialized (Claude Haiku)")

    async def classify(self, state: dict, callbacks: list = None) -> OrchestratorPlan:
        """
        Extract intent and parameters for the deterministic graph.

        Returns:
            OrchestratorPlan. Only its intent signal and params are used by
            the graph; the graph does not let the LLM control runtime edges.
        """
        context = self._build_plan_context(state)
        invoke_config = {"callbacks": callbacks} if callbacks else {}

        try:
            structured_llm = self.llm.with_structured_output(OrchestratorPlan)
            plan = await structured_llm.ainvoke([
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=context),
            ], config=invoke_config)
            logger.info(
                f"[{state.get('request_id')}] Orchestrator plan: "
                f"{[s.agent for s in plan.steps]}"
            )
        except Exception as e:
            logger.warning(f"Orchestrator plan failed ({e}), using fallback")
            plan = create_fallback_plan(state)

        return enforce_plan_rules(plan, state)

    # ── Context builders ──────────────────────────────────────────────────────

    def _build_plan_context(self, state: dict) -> str:
        """
        Build the context passed to Claude.
        """
        query = state.get("user_query", "")
        language = state.get("user_language", "en")
        user_memory = state.get("user_memory")

        lines = [
            f"USER QUERY: {query}",
            f"LANGUAGE: {language}",
        ]

        # Add user profile context when memory is available.
        if user_memory:
            visited = user_memory.get("visited_places", [])
            interests = user_memory.get("interests", [])
            if visited:
                lines.append(f"USER VISITED BEFORE: {', '.join(visited[:5])}")
            if interests:
                lines.append(f"USER INTERESTS: {', '.join(interests[:5])}")

        lines.append(
            "\nCreate a plan to answer this travel query. "
            "Identify the Georgian region from the query and use it as "
            "the 'region' parameter for search_agent."
        )

        return "\n".join(lines)
