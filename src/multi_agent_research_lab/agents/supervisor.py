"""Supervisor / router — decides which worker should run next and when to stop."""

import json
import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

SUPERVISOR_SYSTEM_PROMPT = """\
You are a Supervisor Agent that routes tasks to specialist workers.

Available workers:
- researcher: Searches for sources and creates research notes.
- analyst: Analyzes research notes into structured insights.
- writer: Synthesizes everything into a final answer.

Routing rules:
1. If no research notes exist yet → route to "researcher".
2. If research notes exist but no analysis → route to "analyst".
3. If analysis exists but no final answer → route to "writer".
4. If final answer exists → route to "done".

Respond with ONLY a JSON object: {"next": "<worker_name>", "reason": "<brief reason>"}
Valid values for "next": "researcher", "analyst", "writer", "done"."""

# Deterministic fallback order when LLM is unavailable or returns garbage
_FALLBACK_SEQUENCE = ["researcher", "analyst", "writer", "done"]


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self) -> None:
        settings = get_settings()
        self._llm = LLMClient(temperature=0.0)
        self._max_iterations = settings.max_iterations

    def run(self, state: ResearchState) -> ResearchState:
        """Update ``state.route_history`` with the next route."""

        with trace_span("supervisor_agent", {"iteration": state.iteration}) as span:
            # Guardrail: max iterations
            if state.iteration >= self._max_iterations:
                logger.warning("Max iterations (%d) reached — forcing done",
                               self._max_iterations)
                state.record_route("done")
                state.add_trace_event("supervisor_max_iter", {
                    "iteration": state.iteration,
                })
                return state

            # Try LLM-based routing
            next_route = self._route_with_llm(state)

            # Fallback: deterministic routing if LLM fails
            if next_route is None:
                next_route = self._deterministic_route(state)
                logger.info("Supervisor using deterministic fallback: %s", next_route)

            state.record_route(next_route)

            state.agent_results.append(AgentResult(
                agent=AgentName.SUPERVISOR,
                content=f"Routed to: {next_route}",
                metadata={"iteration": state.iteration, "next": next_route},
            ))
            state.add_trace_event("supervisor_route", {
                "next": next_route,
                "iteration": state.iteration,
                "duration": span.get("duration_seconds"),
            })

        logger.info("Supervisor routed to: %s (iteration %d)", next_route, state.iteration)
        return state

    def _route_with_llm(self, state: ResearchState) -> str | None:
        """Ask the LLM for a routing decision. Returns None on failure."""
        try:
            status_parts = [
                f"Query: {state.request.query}",
                f"Iteration: {state.iteration}",
                f"Has research notes: {state.research_notes is not None}",
                f"Has analysis notes: {state.analysis_notes is not None}",
                f"Has final answer: {state.final_answer is not None}",
                f"Sources count: {len(state.sources)}",
                f"Errors: {state.errors or 'none'}",
            ]
            user_prompt = "\n".join(status_parts)

            response = self._llm.complete(SUPERVISOR_SYSTEM_PROMPT, user_prompt)
            parsed = json.loads(response.content)
            next_route = parsed.get("next", "").lower().strip()

            valid_routes = {"researcher", "analyst", "writer", "done"}
            if next_route in valid_routes:
                return next_route

            logger.warning("Supervisor LLM returned invalid route: %s", next_route)
            return None

        except Exception:
            logger.exception("Supervisor LLM routing failed — will use fallback")
            return None

    def _deterministic_route(self, state: ResearchState) -> str:
        """Rule-based fallback routing when LLM is unavailable."""
        if state.research_notes is None:
            return "researcher"
        if state.analysis_notes is None:
            return "analyst"
        if state.final_answer is None:
            return "writer"
        return "done"
