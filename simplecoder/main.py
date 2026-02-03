import logging
from dataclasses import dataclass
from typing import Optional, Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from simplecoder.agent import Agent


logging.basicConfig(level=logging.ERROR)
console = Console()


@dataclass
class InteractiveConfig:
    """Runtime-configurable settings for interactive mode."""
    model: str = "gemini/gemini-3-flash-preview"
    max_iterations: int = 10
    verbose: bool = False
    use_rag: Optional[bool] = None      # None = use LLM suggestion
    use_planning: Optional[bool] = None  # None = use LLM suggestion


def show_welcome(con: Console):
    """Display welcome message."""
    con.print(Panel(
        "[bold cyan]SimpleCoder Agent[/bold cyan]\n\n"
        "Type your requests and I'll help you code.\n"
        r"End with [bold]\ [/bold]to open settings menu." "\n"
        "Type [bold]help[/bold] for commands, [bold]exit[/bold] to quit.",
        border_style="cyan"
    ))


def show_help(con: Console):
    """Display help information."""
    con.print(Panel(
        "[bold]Commands:[/bold]\n"
        r"  [cyan]<task>[/cyan]      - Run task (features auto-suggested by AI)" "\n"
        r"  [cyan]<task>\ [/cyan]    - Open settings before running task" "\n"
        r"  [cyan]\ [/cyan]          - Open settings menu only" "\n"
        "  [cyan]help[/cyan]        - Show this help\n"
        "  [cyan]exit/quit/q[/cyan] - Exit the program\n\n"
        "[bold]Features:[/bold]\n"
        "  - RAG: Semantic code search (AI suggests when useful)\n"
        "  - Planning: Task decomposition (AI suggests for complex tasks)",
        title="[bold green]Help[/bold green]",
        border_style="green"
    ))


def show_options_menu(con: Console, task: str, config: InteractiveConfig) -> Optional[InteractiveConfig]:
    """Display and handle the options menu."""
    # Snapshot so Cancel can restore — the loop mutates config in-place
    original = InteractiveConfig(
        model=config.model,
        max_iterations=config.max_iterations,
        verbose=config.verbose,
        use_rag=config.use_rag,
        use_planning=config.use_planning,
    )

    while True:
        # Display status with tri-state: ON / auto / OFF
        if config.use_rag is True:
            rag_status = "[green]ON[/green]"
        elif config.use_rag is None:
            rag_status = "[dim]auto[/dim]"
        else:
            rag_status = "[red]OFF[/red]"

        if config.use_planning is True:
            plan_status = "[green]ON[/green]"
        elif config.use_planning is None:
            plan_status = "[dim]auto[/dim]"
        else:
            plan_status = "[red]OFF[/red]"

        model_short = config.model.replace("gemini/", "")

        menu = f"""[bold]Current Settings:[/bold]
  1. RAG (semantic search): {rag_status}
  2. Planning (task decomposition): {plan_status}
  3. Model: [cyan]{model_short}[/cyan]
  4. Max iterations: [cyan]{config.max_iterations}[/cyan]

  [green]5. Run task[/green] | [yellow]0. Cancel[/yellow]

[dim]Tip: Set to 'auto' to let AI suggest features[/dim]"""

        if task:
            con.print(Panel(f"[dim]Task: {task}[/dim]\n\n{menu}", title="[bold blue]Settings[/bold blue]", border_style="blue"))
        else:
            con.print(Panel(menu, title="[bold blue]Settings[/bold blue]", border_style="blue"))

        choice = Prompt.ask("Select option", default="5")

        if choice == "0":
            # Restore everything to what it was before the menu opened
            config.model = original.model
            config.max_iterations = original.max_iterations
            config.use_rag = original.use_rag
            config.use_planning = original.use_planning
            return None
        elif choice == "1":
            # Cycle: None (auto) -> True (on) -> False (off) -> None
            if config.use_rag is None:
                config.use_rag = True
            elif config.use_rag:
                config.use_rag = False
            else:
                config.use_rag = None
        elif choice == "2":
            if config.use_planning is None:
                config.use_planning = True
            elif config.use_planning:
                config.use_planning = False
            else:
                config.use_planning = None
        elif choice == "3":
            models = ["gemini/gemini-3-flash-preview", "gemini/gemini-3-pro-preview", "gemini/gemini-2.5-flash-lite", "gemini/gemini-2.5-flash", "gemini/gemini-2.5-pro"]
            con.print("\n[bold]Available models:[/bold]")
            for i, m in enumerate(models, 1):
                con.print(f"  {i}. {m.replace('gemini/', '')}")
            model_choice = Prompt.ask("Select model", default="1")
            try:
                idx = int(model_choice) - 1
                if 0 <= idx < len(models):
                    config.model = models[idx]
            except ValueError:
                pass
        elif choice == "4":
            try:
                new_iters = int(Prompt.ask("Max iterations", default=str(config.max_iterations)))
                if 1 <= new_iters <= 50:
                    config.max_iterations = new_iters
            except ValueError:
                pass
        elif choice == "5":
            return config

        con.print()


def apply_config_to_agent(agent: Any, config: InteractiveConfig, con: Console) -> bool:
    """Apply configuration changes to the agent dynamically."""
    try:
        agent.model = config.model
        agent.max_iterations = config.max_iterations

        con.print("[green]Settings applied:[/green]")
        con.print(f"  Model: {config.model.replace('gemini/', '')}")
        con.print(f"  Max iterations: {config.max_iterations}")
        return True
    except Exception:
        con.print("[dim]Settings stored, will apply on next run[/dim]")
        return False


def display_features_in_use(con: Console, use_rag: bool, use_planning: bool):
    """Show which features are active."""
    features = []
    if use_rag:
        features.append("[cyan]RAG[/cyan]")
    if use_planning:
        features.append("[magenta]Planning[/magenta]")

    if features:
        con.print(f"[dim]Using: {', '.join(features)}[/dim]")


def detect_intent(task: str) -> dict:
    """Detect intent from task using simple keyword matching.

    Returns dict with suggested features based on task keywords.
    """
    task_lower = task.lower()
    suggestions = {"use_rag": False, "use_planning": False}

    # Keywords that suggest RAG (code search) would help
    rag_keywords = ["find", "search", "where", "locate", "look for", "what is",
                    "how does", "understand", "explain", "structure", "show me"]

    # Keywords that suggest planning would help
    planning_keywords = ["build", "create app", "implement", "develop", "make a",
                        "set up", "scaffold", "generate", "design"]

    for kw in rag_keywords:
        if kw in task_lower:
            suggestions["use_rag"] = True
            break

    for kw in planning_keywords:
        if kw in task_lower:
            suggestions["use_planning"] = True
            break

    return suggestions


def main():
    """Entry point for the simple coder agent."""
    cli()


@click.command()
@click.option(
    "--model",
    default="gemini/gemini-3-flash-preview",
    help="LLM model to use"
)
@click.option(
    "--max-iterations",
    default=10,
    type=int,
    help="Maximum number of ReAct iterations"
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose output"
)
@click.option(
    "--interactive/--no-interactive",
    default=True,
    help="Run in interactive mode"
)
@click.option(
    "--use-planning",
    is_flag=True,
    help="Enable planning and task decomposition"
)
@click.option(
    "--use-rag",
    is_flag=True,
    help="Enable RAG"
)
@click.option(
    "--rag-embedder",
    default="gemini/gemini-embedding-001",
    help="Embedding model for RAG"
)
@click.option(
    "--rag-index-pattern",
    default="**/*.py",
    help="File pattern for RAG"
)
@click.option(
    "--output-dir",
    default="output",
    type=click.Path(),
    help="Output directory for generated files (default: output/)"
)
@click.argument(
    "task",
    required=False
)
def cli(
    model: str,
    max_iterations: int,
    verbose: bool,
    interactive: bool,
    use_planning: bool,
    use_rag: bool,
    rag_embedder: str,
    rag_index_pattern: str,
    output_dir: str,
    task: str | None
) -> None:
    """A simple coding agent."""
    agent = Agent(
        model=model,
        max_iterations=max_iterations,
        verbose=verbose,
        use_planning=use_planning,
        use_rag=use_rag,
        rag_embedder=rag_embedder,
        rag_index_pattern=rag_index_pattern,
        output_dir=output_dir
    )

    if task:
        response = agent.run(task)
        console.print(Panel(Markdown(response), title="[bold green]Agent Response[/bold green]", border_style="green"))
        return

    if interactive:
        show_welcome(console)
        config = InteractiveConfig(
            model=model,
            max_iterations=max_iterations,
            verbose=verbose,
            use_rag=True if use_rag else None,
            use_planning=True if use_planning else None
        )
        console.print()

        while True:
            try:
                user_input = Prompt.ask("\n[bold blue]You[/bold blue]")
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Goodbye![/yellow]")
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                console.print("[yellow]Goodbye![/yellow]")
                break

            if user_input.lower() == "help":
                show_help(console)
                continue

            # Menu trigger (ends with \)
            if user_input.endswith("\\"):
                task = user_input[:-1].strip()
                old_model, old_iters = config.model, config.max_iterations

                result = show_options_menu(console, task, config)
                if result is None:
                    continue
                config = result

                # Apply settings if changed
                if old_model != config.model or old_iters != config.max_iterations:
                    apply_config_to_agent(agent, config, console)

                if not task:
                    console.print("[green]Settings updated[/green]")
                    continue
            else:
                task = user_input

            # Determine features to use (auto-detect if not explicitly set)
            intent = detect_intent(task)

            if config.use_rag is not None:
                use_rag_now = config.use_rag
            else:
                use_rag_now = intent["use_rag"]

            if config.use_planning is not None:
                use_planning_now = config.use_planning
            else:
                use_planning_now = intent["use_planning"]

            # Initialize RAG/planner on-demand if needed
            if use_rag_now and agent.rag is None:
                console.print("[dim]Initializing RAG (semantic code search)...[/dim]")
                from simplecoder.rag import RAG
                agent.rag = RAG(
                    embedder=agent.rag_embedder,
                    index_pattern=agent.rag_index_pattern,
                    verbose=agent.verbose
                )

            if use_planning_now and agent.planner is None:
                console.print("[dim]Initializing planner...[/dim]")
                from simplecoder.planner import TaskPlanner
                agent.planner = TaskPlanner(model="gemini/gemini-3-pro-preview", verbose=agent.verbose)

            # Temporarily set agent flags
            original_rag, original_planning = agent.use_rag, agent.use_planning
            agent.use_rag = use_rag_now
            agent.use_planning = use_planning_now

            # Show active features
            display_features_in_use(console, use_rag_now, use_planning_now)

            try:
                response = agent.run(task)
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                continue
            finally:
                agent.use_rag = original_rag
                agent.use_planning = original_planning

            console.print()
            console.print(Panel(
                Markdown(response),
                title="[bold green]Agent[/bold green]",
                border_style="green"
            ))


if __name__ == "__main__":
    main()
