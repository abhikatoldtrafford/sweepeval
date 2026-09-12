"""``sweepeval compare`` (spec §4.1, I6).

Pure and offline: two run directories in, a verdict out. Nothing is sent and
no credentials are needed, which is what makes it safe to point at a
colleague's committed run.

Exit codes are the point of the verb in CI: ``0`` comparable, ``1`` refused on
a hard key. A refusal is a real outcome, not an error, so it does not exit 2.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from sweepeval.report.compare import compare_runs

console = Console()


def compare_command(
    run_a: Path = typer.Argument(..., help="A run directory or its manifest.json."),
    run_b: Path = typer.Argument(..., help="The run to compare it against."),
) -> None:
    """Check whether two stored runs may be compared."""
    try:
        result = compare_runs(run_a, run_b)
    except FileNotFoundError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error

    if result.ok and not result.warnings:
        console.print(f"[green]{result.explain()}[/green]")
        raise typer.Exit(code=0)

    style = "yellow" if result.ok else "red"
    console.print(f"[{style}]{result.explain()}[/{style}]")
    raise typer.Exit(code=0 if result.ok else 1)
