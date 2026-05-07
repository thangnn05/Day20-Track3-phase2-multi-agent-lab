"""Tests for agent implementations.

Verifies that agents can be instantiated and have correct interfaces.
Integration tests (requiring API keys) are skipped when keys are absent.
"""

from unittest.mock import patch

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMResponse


def _make_state(notes: str | None = None, analysis: str | None = None,
                answer: str | None = None) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.research_notes = notes
    state.analysis_notes = analysis
    state.final_answer = answer
    if notes:
        state.sources = [
            SourceDocument(title="Test Source", snippet="Test snippet", url="https://example.com")
        ]
    return state


_MOCK_LLM_RESPONSE = LLMResponse(
    content="Mock LLM response content for testing.",
    input_tokens=100,
    output_tokens=50,
    cost_usd=0.0001,
)


class TestSupervisorAgent:
    def test_instantiation(self) -> None:
        with patch("multi_agent_research_lab.agents.supervisor.LLMClient"):
            agent = SupervisorAgent()
            assert agent.name == "supervisor"

    def test_deterministic_routing_no_notes(self) -> None:
        """Without research notes, supervisor routes to researcher."""
        with patch("multi_agent_research_lab.agents.supervisor.LLMClient") as mock_cls:
            mock_cls.return_value.complete.side_effect = Exception("No API key")
            agent = SupervisorAgent()
            state = _make_state()
            result = agent.run(state)
            assert result.route_history[-1] == "researcher"

    def test_deterministic_routing_has_notes(self) -> None:
        """With research notes but no analysis, routes to analyst."""
        with patch("multi_agent_research_lab.agents.supervisor.LLMClient") as mock_cls:
            mock_cls.return_value.complete.side_effect = Exception("No API key")
            agent = SupervisorAgent()
            state = _make_state(notes="Some research notes")
            result = agent.run(state)
            assert result.route_history[-1] == "analyst"

    def test_deterministic_routing_has_analysis(self) -> None:
        """With notes and analysis but no answer, routes to writer."""
        with patch("multi_agent_research_lab.agents.supervisor.LLMClient") as mock_cls:
            mock_cls.return_value.complete.side_effect = Exception("No API key")
            agent = SupervisorAgent()
            state = _make_state(notes="Notes", analysis="Analysis")
            result = agent.run(state)
            assert result.route_history[-1] == "writer"

    def test_deterministic_routing_done(self) -> None:
        """With everything filled, routes to done."""
        with patch("multi_agent_research_lab.agents.supervisor.LLMClient") as mock_cls:
            mock_cls.return_value.complete.side_effect = Exception("No API key")
            agent = SupervisorAgent()
            state = _make_state(notes="Notes", analysis="Analysis", answer="Answer")
            result = agent.run(state)
            assert result.route_history[-1] == "done"

    def test_max_iterations_guard(self) -> None:
        """Supervisor forces 'done' when max iterations is reached."""
        with patch("multi_agent_research_lab.agents.supervisor.LLMClient"):
            agent = SupervisorAgent()
            state = _make_state()
            state.iteration = agent._max_iterations  # at the limit
            result = agent.run(state)
            assert result.route_history[-1] == "done"


class TestResearcherAgent:
    def test_instantiation(self) -> None:
        with patch("multi_agent_research_lab.agents.researcher.LLMClient"):
            with patch("multi_agent_research_lab.agents.researcher.SearchClient"):
                agent = ResearcherAgent()
                assert agent.name == "researcher"

    def test_run_populates_state(self) -> None:
        with patch("multi_agent_research_lab.agents.researcher.LLMClient") as mock_llm_cls:
            with patch("multi_agent_research_lab.agents.researcher.SearchClient") as mock_search_cls:
                mock_llm_cls.return_value.complete.return_value = _MOCK_LLM_RESPONSE
                mock_search_cls.return_value.search.return_value = [
                    SourceDocument(title="Source 1", snippet="Snippet 1"),
                ]
                agent = ResearcherAgent()
                state = _make_state()
                result = agent.run(state)
                assert result.research_notes is not None
                assert len(result.sources) == 1
                assert len(result.agent_results) == 1


class TestAnalystAgent:
    def test_run_populates_analysis(self) -> None:
        with patch("multi_agent_research_lab.agents.analyst.LLMClient") as mock_cls:
            mock_cls.return_value.complete.return_value = _MOCK_LLM_RESPONSE
            agent = AnalystAgent()
            state = _make_state(notes="Some research notes")
            result = agent.run(state)
            assert result.analysis_notes is not None

    def test_run_without_notes_records_error(self) -> None:
        with patch("multi_agent_research_lab.agents.analyst.LLMClient"):
            agent = AnalystAgent()
            state = _make_state()
            result = agent.run(state)
            assert result.analysis_notes is None
            assert len(result.errors) == 1


class TestWriterAgent:
    def test_run_populates_final_answer(self) -> None:
        with patch("multi_agent_research_lab.agents.writer.LLMClient") as mock_cls:
            mock_cls.return_value.complete.return_value = _MOCK_LLM_RESPONSE
            agent = WriterAgent()
            state = _make_state(notes="Notes", analysis="Analysis")
            result = agent.run(state)
            assert result.final_answer is not None


class TestCriticAgent:
    def test_run_produces_review(self) -> None:
        with patch("multi_agent_research_lab.agents.critic.LLMClient") as mock_cls:
            mock_cls.return_value.complete.return_value = _MOCK_LLM_RESPONSE
            agent = CriticAgent()
            state = _make_state(notes="N", analysis="A", answer="Final answer here")
            result = agent.run(state)
            assert len(result.agent_results) == 1
            assert result.agent_results[0].agent == "critic"

    def test_run_without_answer_records_error(self) -> None:
        with patch("multi_agent_research_lab.agents.critic.LLMClient"):
            agent = CriticAgent()
            state = _make_state()
            result = agent.run(state)
            assert len(result.errors) == 1
