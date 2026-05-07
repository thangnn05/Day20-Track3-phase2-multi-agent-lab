"""LangGraph workflow — orchestrates the multi-agent research pipeline."""

import logging
import time
from typing import Any

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)


def _state_to_dict(state: ResearchState) -> dict[str, Any]:
    """Convert ResearchState to dict for LangGraph."""
    return state.model_dump()


def _dict_to_state(data: dict[str, Any]) -> ResearchState:
    """Convert LangGraph dict back to ResearchState."""
    return ResearchState.model_validate(data)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in ``agents/``.
    """

    def __init__(self, enable_critic: bool = False) -> None:
        self._enable_critic = enable_critic
        self._supervisor = SupervisorAgent()
        self._researcher = ResearcherAgent()
        self._analyst = AnalystAgent()
        self._writer = WriterAgent()
        self._critic = CriticAgent() if enable_critic else None

    def build(self) -> StateGraph:
        """Create a LangGraph graph with supervisor routing."""

        # Use a simple dict-based state for LangGraph
        graph = StateGraph(dict)

        # --- Node functions (wrap agent.run) ---
        def supervisor_node(data: dict) -> dict:
            state = _dict_to_state(data)
            state = self._supervisor.run(state)
            return _state_to_dict(state)

        def researcher_node(data: dict) -> dict:
            state = _dict_to_state(data)
            state = self._researcher.run(state)
            return _state_to_dict(state)

        def analyst_node(data: dict) -> dict:
            state = _dict_to_state(data)
            state = self._analyst.run(state)
            return _state_to_dict(state)

        def writer_node(data: dict) -> dict:
            state = _dict_to_state(data)
            state = self._writer.run(state)
            return _state_to_dict(state)

        # Add nodes
        graph.add_node("supervisor", supervisor_node)
        graph.add_node("researcher", researcher_node)
        graph.add_node("analyst", analyst_node)
        graph.add_node("writer", writer_node)

        if self._enable_critic and self._critic:
            def critic_node(data: dict) -> dict:
                state = _dict_to_state(data)
                state = self._critic.run(state)
                return _state_to_dict(state)
            graph.add_node("critic", critic_node)

        # --- Routing logic ---
        def route_after_supervisor(data: dict) -> str:
            """Read the last route from route_history to decide next node."""
            route_history = data.get("route_history", [])
            if not route_history:
                return END
            last_route = route_history[-1]
            if last_route == "done":
                if self._enable_critic and data.get("final_answer") and not any(
                    r.get("agent") == "critic" for r in data.get("agent_results", [])
                ):
                    return "critic"
                return END
            if last_route in ("researcher", "analyst", "writer"):
                return last_route
            return END

        # --- Edges ---
        graph.set_entry_point("supervisor")

        conditional_map: dict[str, str] = {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            END: END,
        }
        if self._enable_critic:
            conditional_map["critic"] = "critic"

        graph.add_conditional_edges("supervisor", route_after_supervisor, conditional_map)

        # After each worker, loop back to supervisor for next decision
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")

        if self._enable_critic:
            graph.add_edge("critic", END)

        return graph

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""

        settings = get_settings()
        timeout = settings.timeout_seconds

        with trace_span("multi_agent_workflow", {"query": state.request.query}) as span:
            logger.info("Starting multi-agent workflow for: %s", state.request.query)

            graph = self.build()
            compiled = graph.compile()

            start_time = time.perf_counter()
            initial_data = _state_to_dict(state)

            try:
                # Run the graph with recursion limit as safeguard
                result = compiled.invoke(
                    initial_data,
                    config={"recursion_limit": settings.max_iterations * 2 + 5},
                )
            except Exception as exc:
                elapsed = time.perf_counter() - start_time
                logger.exception("Workflow failed after %.2fs", elapsed)
                raise AgentExecutionError(
                    f"Multi-agent workflow failed: {exc}"
                ) from exc

            elapsed = time.perf_counter() - start_time
            if elapsed > timeout:
                logger.warning("Workflow exceeded timeout: %.2fs > %ds", elapsed, timeout)

            final_state = _dict_to_state(result)
            final_state.add_trace_event("workflow_complete", {
                "total_duration": elapsed,
                "iterations": final_state.iteration,
                "has_answer": final_state.final_answer is not None,
            })

            span["attributes"]["total_duration"] = elapsed
            span["attributes"]["iterations"] = final_state.iteration

        logger.info("Workflow complete: %d iterations, %.2fs",
                     final_state.iteration, elapsed)
        return final_state
