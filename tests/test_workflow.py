"""Tests for the MultiAgentWorkflow orchestration."""

from unittest.mock import patch, MagicMock

import pytest

from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMResponse


_MOCK_LLM = LLMResponse(
    content="Mock response", input_tokens=50, output_tokens=25, cost_usd=0.0001,
)

_MOCK_SEARCH = [
    SourceDocument(title="Mock Source", snippet="Mock snippet", url="https://example.com"),
]


def _patch_all_services():
    """Patch LLMClient and SearchClient across all agent modules."""
    patches = [
        patch("multi_agent_research_lab.agents.supervisor.LLMClient"),
        patch("multi_agent_research_lab.agents.researcher.LLMClient"),
        patch("multi_agent_research_lab.agents.researcher.SearchClient"),
        patch("multi_agent_research_lab.agents.analyst.LLMClient"),
        patch("multi_agent_research_lab.agents.writer.LLMClient"),
        patch("multi_agent_research_lab.agents.critic.LLMClient"),
    ]
    return patches


class TestMultiAgentWorkflow:
    def test_build_returns_graph(self) -> None:
        """Workflow.build() should return a StateGraph."""
        with patch("multi_agent_research_lab.agents.supervisor.LLMClient"), \
             patch("multi_agent_research_lab.agents.researcher.LLMClient"), \
             patch("multi_agent_research_lab.agents.researcher.SearchClient"), \
             patch("multi_agent_research_lab.agents.analyst.LLMClient"), \
             patch("multi_agent_research_lab.agents.writer.LLMClient"):
            workflow = MultiAgentWorkflow()
            graph = workflow.build()
            assert graph is not None

    def test_workflow_end_to_end_with_mocks(self) -> None:
        """Full workflow run with mocked LLM and search services."""
        import json

        # Configure supervisor to return proper JSON routing decisions
        supervisor_responses = iter([
            LLMResponse(content=json.dumps({"next": "researcher", "reason": "need research"}),
                        input_tokens=50, output_tokens=20, cost_usd=0.0001),
            LLMResponse(content=json.dumps({"next": "analyst", "reason": "need analysis"}),
                        input_tokens=50, output_tokens=20, cost_usd=0.0001),
            LLMResponse(content=json.dumps({"next": "writer", "reason": "need writing"}),
                        input_tokens=50, output_tokens=20, cost_usd=0.0001),
            LLMResponse(content=json.dumps({"next": "done", "reason": "complete"}),
                        input_tokens=50, output_tokens=20, cost_usd=0.0001),
        ])

        def supervisor_complete(system, user):
            return next(supervisor_responses)

        with patch("multi_agent_research_lab.agents.supervisor.LLMClient") as sup_llm, \
             patch("multi_agent_research_lab.agents.researcher.LLMClient") as res_llm, \
             patch("multi_agent_research_lab.agents.researcher.SearchClient") as search_cls, \
             patch("multi_agent_research_lab.agents.analyst.LLMClient") as ana_llm, \
             patch("multi_agent_research_lab.agents.writer.LLMClient") as wri_llm:

            sup_llm.return_value.complete.side_effect = supervisor_complete
            res_llm.return_value.complete.return_value = _MOCK_LLM
            search_cls.return_value.search.return_value = _MOCK_SEARCH
            ana_llm.return_value.complete.return_value = _MOCK_LLM
            wri_llm.return_value.complete.return_value = _MOCK_LLM

            workflow = MultiAgentWorkflow()
            state = ResearchState(
                request=ResearchQuery(query="Test multi-agent systems")
            )
            result = workflow.run(state)

            assert result.research_notes is not None
            assert result.analysis_notes is not None
            assert result.final_answer is not None
            assert "done" in result.route_history
            assert result.iteration >= 4  # supervisor ran at least 4 times
