"""Critic agent — optional fact-checking and quality review."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

CRITIC_SYSTEM_PROMPT = """\
You are a Critic Agent. Your job is to review a final answer for quality and accuracy.

Evaluate the response on these criteria:
1. **Citation Coverage**: Are claims backed by referenced sources? Flag unsupported claims.
2. **Accuracy**: Do the conclusions follow logically from the evidence?
3. **Completeness**: Does the answer address the original query fully?
4. **Clarity**: Is the writing clear and well-structured for the target audience?
5. **Hallucination Risk**: Flag any statements that appear fabricated or overly specific \
   without source support.

Output a structured review with:
- Overall quality score (1-10)
- List of specific issues found (if any)
- Suggested improvements (brief, actionable)

Be constructive but honest. If the answer is good, say so."""


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def __init__(self) -> None:
        self._llm = LLMClient(temperature=0.1)

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings to trace."""

        with trace_span("critic_agent") as span:
            if not state.final_answer:
                state.errors.append("Critic: no final answer to review.")
                logger.warning("Critic invoked without a final answer")
                return state

            # Build review context
            source_titles = "\n".join(
                f"- {s.title}" for s in state.sources
            ) or "No sources listed."

            user_prompt = (
                f"Original Query: {state.request.query}\n"
                f"Target Audience: {state.request.audience}\n\n"
                f"Sources Used:\n{source_titles}\n\n"
                f"Final Answer to Review:\n{state.final_answer}"
            )

            response = self._llm.complete(CRITIC_SYSTEM_PROMPT, user_prompt)

            state.agent_results.append(AgentResult(
                agent=AgentName.CRITIC,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            ))
            state.add_trace_event("critic_done", {
                "review_length": len(response.content),
                "duration": span.get("duration_seconds"),
            })

        logger.info("Critic review complete (%d chars)", len(response.content))
        return state
