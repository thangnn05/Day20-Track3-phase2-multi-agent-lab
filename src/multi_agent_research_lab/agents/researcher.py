"""Researcher agent — collects sources and creates concise research notes."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

RESEARCHER_SYSTEM_PROMPT = """\
You are a Research Agent. Your job is to synthesize search results into clear, \
well-organized research notes.

Given a research query and a set of source documents, produce:
1. A summary of key findings from the sources (bullet points).
2. Areas of agreement and disagreement across sources.
3. Gaps or questions that remain unanswered.

Cite sources by their title in brackets, e.g. [Research Paper: ...].
Keep notes concise but information-dense. Target 300-500 words."""


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self) -> None:
        self._search = SearchClient()
        self._llm = LLMClient(temperature=0.2)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.sources`` and ``state.research_notes``."""

        with trace_span("researcher_agent", {"query": state.request.query}) as span:
            # Step 1: Search for sources
            logger.info("Researcher searching for: %s", state.request.query)
            sources = self._search.search(
                state.request.query,
                max_results=state.request.max_sources,
            )
            state.sources = sources

            # Step 2: Format source snippets for LLM
            source_text = "\n\n".join(
                f"[{i+1}] {s.title}\nURL: {s.url or 'N/A'}\n{s.snippet}"
                for i, s in enumerate(sources)
            )

            user_prompt = (
                f"Research Query: {state.request.query}\n"
                f"Target Audience: {state.request.audience}\n\n"
                f"Sources Found ({len(sources)}):\n{source_text}"
            )

            # Step 3: Generate research notes via LLM
            response = self._llm.complete(RESEARCHER_SYSTEM_PROMPT, user_prompt)
            state.research_notes = response.content

            # Record result and trace
            state.agent_results.append(AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={
                    "sources_count": len(sources),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            ))
            state.add_trace_event("researcher_done", {
                "sources": len(sources),
                "notes_length": len(response.content),
                "duration": span.get("duration_seconds"),
            })

        logger.info("Researcher produced %d chars of notes from %d sources",
                     len(state.research_notes), len(sources))
        return state
