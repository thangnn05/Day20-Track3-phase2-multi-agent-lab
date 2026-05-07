"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline that handles the full query in one LLM call."""

    _init()

    def _run_baseline(q: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=q))
        llm = LLMClient(temperature=0.3)
        system_prompt = (
            "You are a helpful research assistant. Answer the following query "
            "comprehensively in about 500 words. Include structure with headings "
            "and cite any claims where possible."
        )
        response = llm.complete(system_prompt, q)
        state.final_answer = response.content
        state.add_trace_event("baseline_complete", {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
        })
        return state

    state, metrics = run_benchmark("single-agent-baseline", query, _run_baseline)
    metrics.notes = "Single LLM call, no agent decomposition"

    console.print(Panel.fit(state.final_answer or "No answer produced.", title="Single-Agent Baseline"))
    _print_metrics(metrics.run_name, metrics.latency_seconds, metrics.estimated_cost_usd)


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    critic: Annotated[bool, typer.Option("--critic", help="Enable critic agent")] = False,
) -> None:
    """Run the multi-agent workflow (Supervisor → Researcher → Analyst → Writer)."""

    _init()

    def _run_multi(q: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=q))
        workflow = MultiAgentWorkflow(enable_critic=critic)
        return workflow.run(state)

    try:
        state, metrics = run_benchmark("multi-agent", query, _run_multi)
    except AgentExecutionError as exc:
        console.print(Panel.fit(str(exc), title="Workflow Error", style="red"))
        raise typer.Exit(code=1) from exc

    metrics.notes = f"Agents: supervisor→researcher→analyst→writer{' →critic' if critic else ''}"

    console.print(Panel.fit(state.final_answer or "No answer produced.", title="Multi-Agent Result"))
    _print_metrics(metrics.run_name, metrics.latency_seconds, metrics.estimated_cost_usd)
    _print_trace_summary(state)


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    critic: Annotated[bool, typer.Option("--critic", help="Enable critic agent")] = False,
    save: Annotated[bool, typer.Option("--save", help="Save report to reports/")] = True,
) -> None:
    """Run both baseline and multi-agent, then produce a comparison report."""

    _init()
    all_metrics = []

    # --- Baseline ---
    console.print("[bold]Running single-agent baseline...[/bold]")

    def _run_baseline(q: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=q))
        llm = LLMClient(temperature=0.3)
        response = llm.complete(
            "You are a helpful research assistant. Answer comprehensively in ~500 words.",
            q,
        )
        state.final_answer = response.content
        state.add_trace_event("baseline_complete", {
            "cost_usd": response.cost_usd,
        })
        return state

    baseline_state, baseline_metrics = run_benchmark("single-agent-baseline", query, _run_baseline)
    baseline_metrics.notes = "Single LLM call"
    all_metrics.append(baseline_metrics)

    # --- Multi-agent ---
    console.print("[bold]Running multi-agent workflow...[/bold]")

    def _run_multi(q: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=q))
        workflow = MultiAgentWorkflow(enable_critic=critic)
        return workflow.run(state)

    multi_state, multi_metrics = run_benchmark("multi-agent", query, _run_multi)
    multi_metrics.notes = f"Supervisor + 3 workers{' + critic' if critic else ''}"
    all_metrics.append(multi_metrics)

    # --- Report ---
    report = render_markdown_report(all_metrics)
    console.print(Panel(report, title="Benchmark Report"))

    if save:
        store = LocalArtifactStore()
        path = store.write_text("benchmark_report.md", report)
        console.print(f"[green]Report saved to {path}[/green]")


def _print_metrics(run_name: str, latency: float, cost: float | None) -> None:
    table = Table(title="Metrics")
    table.add_column("Run")
    table.add_column("Latency (s)", justify="right")
    table.add_column("Est. Cost ($)", justify="right")
    cost_str = f"{cost:.6f}" if cost else "N/A"
    table.add_row(run_name, f"{latency:.2f}", cost_str)
    console.print(table)


def _print_trace_summary(state: ResearchState) -> None:
    if not state.route_history:
        return
    console.print(f"\n[dim]Route history: {' → '.join(state.route_history)}[/dim]")
    console.print(f"[dim]Iterations: {state.iteration} | Errors: {len(state.errors)}[/dim]")
    if state.errors:
        for err in state.errors:
            console.print(f"  [red]• {err}[/red]")


if __name__ == "__main__":
    app()
