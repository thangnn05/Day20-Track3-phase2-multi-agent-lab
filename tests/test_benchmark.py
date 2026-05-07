"""Tests for benchmark and report modules."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark, _estimate_quality
from multi_agent_research_lab.evaluation.report import render_markdown_report


def _dummy_runner(query: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    state.final_answer = (
        "# Test Answer\n\n"
        "This is a comprehensive answer about the topic. "
        "It includes multiple paragraphs and references [Source A] and [Source B].\n\n"
        "## Key Findings\n\n"
        "- Finding one with supporting evidence\n"
        "- Finding two with additional context\n"
        "- Finding three with practical implications\n\n"
        "## Conclusion\n\n"
        "The research demonstrates significant progress in this area."
    )
    return state


class TestRunBenchmark:
    def test_returns_state_and_metrics(self) -> None:
        state, metrics = run_benchmark("test-run", "Test query for benchmark", _dummy_runner)
        assert state.final_answer is not None
        assert metrics.run_name == "test-run"
        assert metrics.latency_seconds >= 0

    def test_measures_latency(self) -> None:
        _, metrics = run_benchmark("latency-test", "Test query here", _dummy_runner)
        assert metrics.latency_seconds >= 0  # runner is near-instant, so >= 0 is valid


class TestQualityEstimation:
    def test_no_answer_gives_zero(self) -> None:
        state = ResearchState(request=ResearchQuery(query="Test query"))
        score = _estimate_quality(state)
        assert score == 0.0

    def test_good_answer_scores_well(self) -> None:
        state = _dummy_runner("Test query for scoring")
        score = _estimate_quality(state)
        assert score is not None
        assert score > 5.0  # structured answer with citations should score well

    def test_errors_reduce_score(self) -> None:
        state = _dummy_runner("Test query")
        score_clean = _estimate_quality(state)
        state.errors = ["error 1", "error 2"]
        score_with_errors = _estimate_quality(state)
        assert score_clean is not None and score_with_errors is not None
        assert score_with_errors < score_clean


class TestRenderReport:
    def test_renders_markdown_table(self) -> None:
        metrics = [BenchmarkMetrics(run_name="baseline", latency_seconds=1.23)]
        report = render_markdown_report(metrics)
        assert "Benchmark Report" in report
        assert "baseline" in report
        assert "1.23" in report

    def test_comparison_analysis_with_two_runs(self) -> None:
        metrics = [
            BenchmarkMetrics(run_name="baseline", latency_seconds=2.0,
                             estimated_cost_usd=0.001, quality_score=5.0),
            BenchmarkMetrics(run_name="multi-agent", latency_seconds=8.0,
                             estimated_cost_usd=0.004, quality_score=8.0),
        ]
        report = render_markdown_report(metrics)
        assert "Comparison Analysis" in report
        assert "Latency" in report
        assert "Cost" in report
        assert "Quality" in report
