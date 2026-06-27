"""CLI interface for Context Router."""

from __future__ import annotations

import asyncio
import json
import logging
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from context_router.config import RouterConfig, load_config, save_config
from context_router.models import PromptAnalysis, RouterResult, RoutingStats, RoutingStrategy
from context_router.router import Router

console = Console()
logger = logging.getLogger("context_router")


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option(
    "--config",
    "config_path",
    default=None,
    help="Path to config file (default: config.yaml)",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Logging level",
)
@click.pass_context
def main(ctx: click.Context, config_path: str | None, log_level: str) -> None:
    """Context Router — Smart AI request router."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path
    setup_logging(log_level)


@main.command()
@click.argument("prompt")
@click.option(
    "--system-prompt",
    "-s",
    default=None,
    help="System prompt to include",
)
@click.option(
    "--provider",
    "-p",
    default=None,
    help="Specific provider to use (bypasses routing)",
)
@click.option(
    "--strategy",
    "-r",
    default=None,
    type=click.Choice([s.value for s in RoutingStrategy]),
    help="Override routing strategy",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def route(
    ctx: click.Context,
    prompt: str,
    system_prompt: str | None,
    provider: str | None,
    strategy: str | None,
    as_json: bool,
) -> None:
    """Route a prompt to the best provider."""
    config_path = ctx.obj.get("config_path")
    config = load_config(config_path)

    if strategy:
        config.routing_strategy = RoutingStrategy(strategy)

    router = Router(config)

    try:
        result = asyncio.run(
            router.route(prompt, system_prompt=system_prompt, provider_name=provider)
        )
    except Exception as e:
        logger.error("Routing failed: %s", e)
        console.print(Panel(f"[red]Error:[/red] {e}", title="Routing Error", border_style="red"))
        sys.exit(1)

    if as_json:
        console.print(json.dumps(result.model_dump(), indent=2))
    else:
        _print_route_result(result)


@main.command()
@click.argument("prompt")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def analyze(ctx: click.Context, prompt: str, as_json: bool) -> None:
    """Analyze a prompt without routing."""
    config_path = ctx.obj.get("config_path")
    config = load_config(config_path)
    router = Router(config)

    analysis = router.analyze(prompt)

    if as_json:
        console.print(json.dumps(analysis.model_dump(), indent=2))
    else:
        _print_analysis(analysis)


@main.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def providers(ctx: click.Context, as_json: bool) -> None:
    """List all configured providers."""
    config_path = ctx.obj.get("config_path")
    config = load_config(config_path)

    if not config.providers:
        console.print("[yellow]No providers configured.[/yellow]")
        return

    table = Table(title="Configured Providers")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Model", style="magenta")
    table.add_column("Speed", style="blue")
    table.add_column("Quality", style="yellow")
    table.add_column("Cost/Token", style="dim")
    table.add_column("Base URL", style="dim")

    for name, provider in config.providers.items():
        table.add_row(
            name,
            provider.type.value,
            provider.model,
            provider.speed.value,
            provider.quality.value,
            f"${provider.cost_per_token:.10f}" if provider.cost_per_token > 0 else "Free",
            provider.base_url,
        )

    console.print(table)


@main.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def stats(ctx: click.Context, as_json: bool) -> None:
    """Show routing statistics."""
    config_path = ctx.obj.get("config_path")
    config = load_config(config_path)
    router = Router(config)

    stats = router.get_stats()

    if as_json:
        console.print(json.dumps(stats.model_dump(), indent=2))
    else:
        _print_stats(stats)


@main.command()
@click.option(
    "--strategy",
    "-r",
    default="hybrid",
    type=click.Choice([s.value for s in RoutingStrategy]),
    help="Routing strategy",
)
@click.option(
    "--output",
    "-o",
    default="config.yaml",
    help="Output config file path",
)
@click.pass_context
def sample_config(ctx: click.Context, strategy: str, output: str) -> None:
    """Generate a sample configuration file."""
    config = RouterConfig(
        routing_strategy=RoutingStrategy(strategy),
        default_provider="fast",
    )
    save_config(config, output)
    console.print(f"[green]Sample config written to {output}[/green]")


@main.command()
@click.option(
    "--reset",
    is_flag=True,
    help="Reset routing statistics",
)
@click.pass_context
def reset(ctx: click.Context, reset: bool) -> None:
    """Reset router state."""
    if reset:
        config_path = ctx.obj.get("config_path")
        config = load_config(config_path)
        router = Router(config)
        router.reset_stats()
        console.print("[green]Routing statistics reset.[/green]")


def _print_route_result(result: RouterResult) -> None:
    """Print a routing result in a nice format."""
    console.print(Panel(
        f"[bold]{result.provider_name}[/bold] ({result.model})\n\n"
        f"Strategy: {result.routing_strategy}\n"
        f"Reason: {result.reason}\n"
        f"Estimated cost: ${result.estimated_cost:.6f}\n"
        f"Estimated tokens: {result.estimated_tokens}\n"
        f"Latency estimate: {result.latency_estimate}",
        title="Routing Result",
        border_style="green",
    ))

    if result.response:
        console.print(Panel(result.response, title="Response", border_style="blue"))


def _print_analysis(analysis: PromptAnalysis) -> None:
    """Print a prompt analysis."""
    table = Table(title="Prompt Analysis")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Complexity", f"{analysis.complexity:.2f}")
    table.add_row("Task Type", analysis.task_type)
    table.add_row("Estimated Tokens", str(analysis.estimated_tokens))
    table.add_row("Requires Reasoning", str(analysis.requires_reasoning))
    table.add_row("Requires Creativity", str(analysis.requires_creativity))
    table.add_row("Urgency", analysis.urgency)
    table.add_row("Confidence", f"{analysis.confidence:.2f}")

    console.print(table)


def _print_stats(stats: RoutingStats) -> None:
    """Print routing statistics."""
    table = Table(title="Routing Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Requests", str(stats.total_requests))
    table.add_row("Total Tokens", str(stats.total_tokens))
    table.add_row("Total Cost", f"${stats.total_cost:.6f}")
    table.add_row("Avg Cost/Request", f"${stats.average_cost_per_request:.6f}")
    table.add_row("Avg Tokens/Request", f"{stats.average_tokens_per_request:.1f}")

    if stats.provider_counts:
        table.add_row("", "")
        table.add_row("Provider Distribution", "")
        for name, count in stats.provider_counts.items():
            table.add_row(f"  {name}", str(count))

    if stats.strategy_counts:
        table.add_row("", "")
        table.add_row("Strategy Distribution", "")
        for name, count in stats.strategy_counts.items():
            table.add_row(f"  {name}", str(count))

    console.print(table)


if __name__ == "__main__":
    main()
