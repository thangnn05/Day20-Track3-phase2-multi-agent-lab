"""Writer agent — produces final answer from research and analysis notes."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

WRITER_SYSTEM_PROMPT = """\
You are a Writer Agent. Your job is to synthesize research notes and analysis into \
a clear, well-structured final response.

Guidelines:
1. Write for the specified target audience.
2. Use clear headings and structure.
3. Include citations referencing the source titles, e.g. [Source Title].
4. Start with a concise executive summary (2-3 sentences).
5. Cover key findings, analysis, and practical implications.
6. End with a brief conclusion or forward-looking statement.

Target approximately 500 words unless the query specifies otherwise. \
Prioritize clarity and accuracy over length."""


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self) -> None:
        self._llm = LLMClient(temperature=0.4)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.final_answer``."""

        with trace_span("writer_agent") as span:
            # Build comprehensive context
            sections: list[str] = [
                f"Research Query: {state.request.query}",
                f"Target Audience: {state.request.audience}",
            ]

            if state.research_notes:
                sections.append(f"\n## Research Notes\n{state.research_notes}")
            if state.analysis_notes:
                sections.append(f"\n## Analysis\n{state.analysis_notes}")
            if state.sources:
                source_list = "\n".join(
                    f"- [{s.title}]({s.url})" if s.url else f"- {s.title}"
                    for s in state.sources
                )
                sections.append(f"\n## Available Sources\n{source_list}")

            user_prompt = "\n".join(sections)

            response = self._llm.complete(WRITER_SYSTEM_PROMPT, user_prompt)
            state.final_answer = response.content

            state.agent_results.append(AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                    "word_count": len(response.content.split()),
                },
            ))
            state.add_trace_event("writer_done", {
                "answer_length": len(response.content),
                "word_count": len(response.content.split()),
                "duration": span.get("duration_seconds"),
            })

        logger.info("Writer produced %d words", len(state.final_answer.split()))
        return state
