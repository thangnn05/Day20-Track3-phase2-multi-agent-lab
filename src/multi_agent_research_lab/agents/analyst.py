"""Analyst agent — turns research notes into structured insights."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

ANALYST_SYSTEM_PROMPT = """\
You are an Analyst Agent. Your job is to turn raw research notes into structured, \
critical analysis.

Given research notes and source information, produce:
1. **Key Claims**: List the 3-5 most important claims with supporting evidence.
2. **Viewpoint Comparison**: Where do sources agree and disagree?
3. **Evidence Quality**: Flag any claims with weak or missing evidence.
4. **Synthesis**: A 2-3 sentence high-level takeaway.

Be analytical and critical. Identify potential biases or gaps. \
Target 200-400 words of structured analysis."""


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self) -> None:
        self._llm = LLMClient(temperature=0.1)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.analysis_notes``."""

        with trace_span("analyst_agent") as span:
            if not state.research_notes:
                state.errors.append("Analyst: no research notes available to analyze.")
                logger.warning("Analyst invoked without research notes")
                return state

            # Build context from research notes and sources
            source_titles = "\n".join(
                f"- {s.title}" for s in state.sources
            ) or "No sources listed."

            user_prompt = (
                f"Research Query: {state.request.query}\n"
                f"Target Audience: {state.request.audience}\n\n"
                f"Sources Referenced:\n{source_titles}\n\n"
                f"Research Notes:\n{state.research_notes}"
            )

            response = self._llm.complete(ANALYST_SYSTEM_PROMPT, user_prompt)
            state.analysis_notes = response.content

            state.agent_results.append(AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            ))
            state.add_trace_event("analyst_done", {
                "analysis_length": len(response.content),
                "duration": span.get("duration_seconds"),
            })

        logger.info("Analyst produced %d chars of analysis", len(state.analysis_notes))
        return state
