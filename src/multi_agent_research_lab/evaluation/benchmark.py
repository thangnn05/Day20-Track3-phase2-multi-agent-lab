"""Benchmark for single-agent vs multi-agent comparison."""

import logging
from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]


def _estimate_total_cost(state: ResearchState) -> float | None:
    """Sum cost_usd from all agent results."""
    costs = [r.metadata.get("cost_usd") for r in state.agent_results if r.metadata.get("cost_usd")]
    # Also check trace events for baseline cost
    for event in state.trace:
        if event.get("payload", {}).get("cost_usd"):
            costs.append(event["payload"]["cost_usd"])
    return sum(costs) if costs else None


def _estimate_quality(state: ResearchState) -> float | None:
    """Heuristic quality score (0-10) based on output characteristics.

    A real implementation would use LLM-as-judge or human evaluation.
    This heuristic checks: has answer, length, sources cited, structure.
    """
    if not state.final_answer:
        return 0.0

    score = 0.0
    answer = state.final_answer

    # Has a non-trivial answer
    word_count = len(answer.split())
    if word_count >= 100:
        score += 2.0
    elif word_count >= 50:
        score += 1.0

    # Length bonus (up to 2 points for ~500 words)
    score += min(2.0, word_count / 250)

    # Has structure (headings, bullet points)
    if any(marker in answer for marker in ["#", "**", "- ", "1."]):
        score += 1.5

    # References sources
    if "[" in answer and "]" in answer:
        score += 1.5

    # Has multiple sections
    paragraphs = [p.strip() for p in answer.split("\n\n") if p.strip()]
    if len(paragraphs) >= 3:
        score += 1.0

    # No errors during execution
    if not state.errors:
        score += 1.0
    else:
        score -= 0.5 * len(state.errors)

    return min(10.0, max(0.0, round(score, 1)))


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, quality, and cost for a given runner."""

    logger.info("Starting benchmark run: %s", run_name)
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    cost = _estimate_total_cost(state)
    quality = _estimate_quality(state)

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 3),
        estimated_cost_usd=cost,
        quality_score=quality,
        notes="",
    )

    logger.info("Benchmark [%s]: latency=%.2fs, cost=$%s, quality=%s",
                run_name, latency,
                f"{cost:.6f}" if cost else "N/A",
                f"{quality:.1f}" if quality else "N/A")

    return state, metrics
