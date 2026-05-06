"""
main.py

Entry point for the stem agent system.

Orchestrates two phases:
1. Stem phase: observes the domain, synthesizes a configuration, and
   generates a specialized agent. Skips transformation if a config
   already exists on disk.
2. Agent phase: runs the specialized agent in an interactive loop,
   accepting tasks from the user until 'exit' is entered.

Design decisions:
- Domain comes from STEM_DOMAIN env var: the domain is an operator
  decision, not a user decision. The user interacts with the specialized
  agent, not with the transformation process.
- main() contains no business logic: it delegates entirely to
  run_stem_phase and run_agent_phase. Each function has a single
  responsibility.
"""

import os

from dotenv import load_dotenv

from agent.react import run
from core.logger import get_logger
from stem.generate import generate, load_existing
from stem.observe import observe
from stem.synthesize import synthesize

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.align import Align
from rich.rule import Rule
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)

load_dotenv()

log = get_logger(__name__)
console = Console()


def render_header():
    header = Text()
    header.append("STEM AGENT\n", style="bold green")
    header.append("Self-specializing AI system", style="dim")

    console.print(Panel(Align.center(header), border_style="green"))


def run_stem_phase(domain: str) -> dict:
    """
    Run the stem transformation phase for the given domain.

    Loads an existing agent config from disk if available, skipping
    the expensive transformation. Otherwise runs observe → synthesize
    → generate and returns the resulting config.

    Args:
        domain: The problem domain, e.g. 'Technical Research'.

    Returns:
        Executable agent configuration dict.
    """
    existing = load_existing(domain)

    if existing:
        console.print(
            Panel(
                f"[green]✔ Loaded existing config[/green]\n[dim]{domain}[/dim]",
                title="STEM Phase",
                border_style="green",
            )
        )
        log.info("Found existing config for '%s'. Skipping transformation.", domain)
        return existing

    console.print(Rule("[bold yellow]STEM Phase: Transformation[/bold yellow]"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Observing domain...", total=3)

        observations = observe(domain)
        progress.update(task, advance=1, description="Synthesizing knowledge...")

        synthesis = synthesize(domain, observations)
        progress.update(task, advance=1, description="Generating agent...")

        result = generate(domain, synthesis, observations)
        progress.update(task, advance=1, description="[bold green]Done![/bold green]")

    console.print(
        Panel(
            "[green]✔ Agent successfully generated[/green]",
            border_style="green",
        )
    )

    return result["config"]


def run_agent_phase(agent_config: dict) -> None:
    """
    Run the specialized agent in an interactive loop.

    Accepts tasks from the user until 'exit' is entered.
    Each task is routed through the ReAct loop and the result
    is printed to stdout.

    Args:
        agent_config: Executable agent configuration dict.
    """
    console.print(Rule("[bold cyan]Agent Ready[/bold cyan]"))

    identity = agent_config.get("identity", "Unknown Agent")
    domain = agent_config.get("domain", "Unknown Domain")

    console.print(
        Panel(
            f"[bold green]{identity}[/bold green]\n"
            f"[dim]Domain:[/dim] {domain}\n"
            f"[dim]Type 'exit' to quit[/dim]",
            title="Agent",
            border_style="cyan",
        )
    )

    try:
        while True:
            task = Prompt.ask("\n[bold cyan]▶ Task[/bold cyan]").strip()

            if task.lower() == "exit":
                console.print("[dim]Shutting down...[/dim]")
                log.info("Shutting down.")
                break

            if not task:
                continue

            with console.status("[bold green]Thinking..."):
                result = run(task, agent_config)

            if not result["accepted"]:
                console.print(
                    Panel(
                        result["answer"],
                        title="[bold yellow]Rejected[/bold yellow]",
                        border_style="yellow",
                    )
                )
            elif result["error"]:
                console.print(
                    Panel(
                        result["answer"],
                        title="[bold red]System Error[/bold red]",
                        border_style="red",
                    )
                )
            else:
                console.print(
                    Panel(
                        Markdown(result["answer"]),
                        title=f"[bold green]Answer[/bold green] [dim]({result['steps']} steps)[/dim]",
                        border_style="green",
                    )
                )

    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted. Shutting down...[/dim]")
        log.info("Interrupted. Shutting down.")


def main() -> None:
    """
    Main entry point for the stem agent system.

    Reads the domain from the STEM_DOMAIN environment variable,
    runs the stem phase to obtain or load a specialized agent config,
    then starts the interactive agent loop.
    """
    console.clear()
    console.rule("[dim]Initializing...[/dim]")

    render_header()

    domain = os.getenv("STEM_DOMAIN")
    if not domain:
        console.print(
            Panel(
                "[red]Error:[/red] STEM_DOMAIN not set in .env",
                border_style="red",
            )
        )
        raise ValueError("STEM_DOMAIN not set in .env")

    agent_config = run_stem_phase(domain)
    run_agent_phase(agent_config)


if __name__ == "__main__":
    main()
