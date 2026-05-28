"""Claude-based request classifier for deterministic graph routing."""

from __future__ import annotations

import logging
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator.fallback_rules import create_fallback_plan, enforce_plan_rules
from orchestrator.prompts import load_prompt
from state.contracts import QueryClassification
from state.models import OrchestratorPlan

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """Claude Haiku based request classifier."""

    def __init__(self):
        from config.settings import get_settings

        settings = get_settings()
        self.llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            api_key=settings.anthropic_api_key,
            max_tokens=1000,
            temperature=0.0,
        )
        self.system_prompt = load_prompt("orchestrator")
        logger.info("OrchestratorAgent classifier initialized (Claude Haiku)")

    async def classify(self, state: dict, callbacks: list = None) -> QueryClassification:
        """Extract intent and parameters for deterministic routing."""
        context = self._build_plan_context(state)
        invoke_config = {"callbacks": callbacks} if callbacks else {}

        try:
            structured_llm = self.llm.with_structured_output(QueryClassification)
            classification = await structured_llm.ainvoke(
                [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=context),
                ],
                config=invoke_config,
            )
            logger.info(
                f"[{state.get('request_id')}] classifier: intent={classification.intent} "
                f"region={classification.region} days={classification.days} pace={classification.pace}"
            )
            return self._attach_legacy_steps(classification, state)
        except Exception as exc:
            logger.warning(f"Classifier failed ({exc}), using fallback heuristics")
            return self._fallback_classification(state)

    def _fallback_classification(self, state: dict) -> QueryClassification:
        """Derive a safe classifier output from fallback plan heuristics."""
        plan = enforce_plan_rules(create_fallback_plan(state), state)
        return self._attach_legacy_steps(self._plan_to_classification(plan, state), state)

    def _attach_legacy_steps(
        self,
        classification: QueryClassification,
        state: dict,
    ) -> QueryClassification:
        """
        Attach legacy step payloads so the current graph can keep working.
        """
        if classification.steps:
            return classification

        intent = classification.intent
        has_current_plan = bool(state.get("has_current_plan"))

        if intent == "PLAN":
            steps = [
                {"agent": "search_agent", "params": {"region": classification.region, "search_query": classification.search_query}, "reason": "classifier: planning request"},
                {"agent": "geo_agent", "params": {"region": classification.region}, "reason": "classifier: geocode for route distances"},
                {"agent": "planning_agent", "params": {"days": classification.days, "pace": classification.pace}, "reason": "classifier: build itinerary"},
                {"agent": "validation_agent", "params": {"pace": classification.pace}, "reason": "classifier: validate itinerary"},
                {"agent": "response_agent", "params": {}, "reason": "classifier: format final response"},
            ]
        elif intent == "REVISE" and has_current_plan:
            steps = [
                {"agent": "search_agent", "params": {"region": classification.region, "search_query": classification.search_query}, "reason": "classifier: plan revision support"},
                {"agent": "response_agent", "params": {}, "reason": "classifier: discuss revision"},
            ]
        else:
            steps = [
                {"agent": "search_agent", "params": {"region": classification.region, "search_query": classification.search_query}, "reason": "classifier: exploration"},
                {"agent": "response_agent", "params": {}, "reason": "classifier: format response"},
            ]

        classification.steps = steps
        classification.estimated_agents = len(steps)
        if not classification.reasoning:
            classification.reasoning = f"{intent} route prepared by classifier"
        return classification

    def _plan_to_classification(
        self,
        plan: OrchestratorPlan,
        state: dict[str, Any],
    ) -> QueryClassification:
        steps = list(getattr(plan, "steps", []) or [])
        params: dict[str, Any] = {}
        agents = set()
        for step in steps:
            agent = step.agent if hasattr(step, "agent") else step.get("agent")
            agents.add(agent)
            step_params = step.params if hasattr(step, "params") else step.get("params", {})
            if step_params:
                params.update(step_params)

        query = state.get("user_query", "")
        has_plan = bool(state.get("has_current_plan"))
        is_revision = has_plan and self._looks_like_revision(query)

        if is_revision:
            intent = "REVISE"
        elif "planning_agent" in agents:
            intent = "PLAN"
        elif "search_agent" in agents:
            intent = "INFO"
        else:
            intent = "CONSULT"

        return QueryClassification(
            intent=intent,
            region=str(params.get("region", "")),
            search_query=str(params.get("search_query", query or "")),
            days=params.get("days"),
            pace=params.get("pace", "moderate"),
            should_ask_clarification=bool(params.get("should_ask_clarification", False)),
            reasoning=str(getattr(plan, "reasoning", "") or ""),
            estimated_agents=int(getattr(plan, "estimated_agents", len(steps) or 0) or 0),
            steps=[
                step.model_dump() if hasattr(step, "model_dump") else dict(step)
                for step in steps
            ],
        )

    @staticmethod
    def _looks_like_revision(query: str) -> bool:
        query_l = (query or "").lower()
        revision_markers = (
            "измен", "помен", "поправ", "скоррект", "обнов", "revise",
            "change", "update", "adjust", "modify", "edit", "rework",
        )
        return any(marker in query_l for marker in revision_markers)

    def _build_plan_context(self, state: dict) -> str:
        query = state.get("user_query", "")
        language = state.get("user_language", "en")
        user_memory = state.get("user_memory")
        has_current_plan = bool(state.get("has_current_plan"))
        current_plan_status = state.get("current_plan_status", "none")

        lines = [
            f"USER QUERY: {query}",
            f"LANGUAGE: {language}",
            f"HAS CURRENT PLAN: {has_current_plan}",
            f"CURRENT PLAN STATUS: {current_plan_status}",
        ]

        if user_memory:
            visited = user_memory.get("visited_places", [])
            interests = user_memory.get("interests", [])
            if visited:
                lines.append(f"USER VISITED BEFORE: {', '.join(visited[:5])}")
            if interests:
                lines.append(f"USER INTERESTS: {', '.join(interests[:5])}")

        lines.append(
            "\nClassify this travel request. "
            "Return the detected Georgian region, normalized search query, "
            "trip days when available, and pace when available. "
            "Use REVISE only when the user is modifying an existing plan."
        )

        return "\n".join(lines)

