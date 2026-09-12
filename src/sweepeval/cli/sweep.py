"""``sweepeval sweep`` and ``sweepeval run`` (spec §4.1, §12.3, I9).

``run`` is the zero-config verb the tool is named for: a URL and a key, and
nothing else. ``sweep`` is the same machinery with the knobs exposed.

The pre-flight confirmation lives here rather than in the engine because it is
the one part that has to know whether a human is watching. The engine takes a
predicate; this module supplies one that prompts on a TTY, honours
``--yes``, and — critically — **declines** when there is no TTY and no
``--yes``, rather than assuming consent. A CI job that starts spending several
thousand requests because nobody was there to say no is exactly the failure
I9 exists to prevent.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console

from sweepeval.execute.budget import BudgetCap, Estimate, render_estimate
from sweepeval.execute.sweep import SweepResult, SweepStatus, asweep_target
from sweepeval.report.sweep import render_sweep

console = Console()


def _confirmer(yes: bool, *, no_input: bool) -> object:
    """Build the pre-flight predicate (§12.3)."""

    def confirm(estimate: Estimate) -> bool:
        console.print("\n[bold]before spending anything[/bold]")
        console.print(render_estimate(estimate))
        console.print()
        if yes:
            console.print("[dim]--yes: proceeding without asking[/dim]")
            return True
        if no_input or not sys.stdin.isatty():
            console.print(
                "[yellow]no terminal to confirm on and --yes was not given, so "
                "nothing was sent.[/yellow]\nRe-run with --yes to accept this "
                "estimate."
            )
            return False
        return typer.confirm("proceed?", default=False)

    return confirm


def _cap(max_requests: int | None) -> BudgetCap | None:
    return BudgetCap(value=max_requests, unit="requests") if max_requests else None


def sweep_command(
    url: str = typer.Argument(..., help="The endpoint to sweep."),
    key: str | None = typer.Option(None, "--key", "-k"),
    profile: str = typer.Option("quick", "--profile", help="quick | standard | deep"),
    runs: int = typer.Option(3, "-n", "--runs"),
    root: Path = typer.Option(Path(".sweepeval"), "--root"),
    seed: int = typer.Option(0, "--seed"),
    authorized: bool = typer.Option(
        False,
        "--i-am-authorized",
        help="Affirm you are authorised to run the security suite against this host.",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Accept the pre-flight estimate without prompting."
    ),
    max_requests: int | None = typer.Option(
        None, "--max-requests", help="Hard cap; the sweep stops between configs."
    ),
    resume: str | None = typer.Option(
        None, "--resume", help="Continue an existing run id."
    ),
) -> None:
    """Discover, plan and sweep every discoverable configuration."""
    result = asyncio.run(
        asweep_target(
            url,
            key=key,
            profile=profile,  # type: ignore[arg-type]
            runs=runs,
            root=str(root),
            seed=seed,
            authorized=authorized,
            confirm=_confirmer(  # type: ignore[arg-type]
                yes, no_input=False, cap=_cap(max_requests)
            ),
            cap=_cap(max_requests),
            resume_run_id=resume,
            on_progress=lambda msg: console.print(f"[dim]{msg}[/dim]"),
        )
    )
    _finish(result)


def run_command(
    url: str = typer.Argument(..., help="The endpoint to evaluate."),
    key: str | None = typer.Option(None, "--key", "-k"),
    profile: str = typer.Option("quick", "--profile", help="quick | standard | deep"),
    runs: int = typer.Option(3, "-n", "--runs"),
    root: Path = typer.Option(Path(".sweepeval"), "--root"),
    seed: int = typer.Option(0, "--seed"),
    authorized: bool = typer.Option(False, "--i-am-authorized"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Zero-config: discover, plan, sweep and report. One URL, one key."""
    sweep_command(
        url=url,
        key=key,
        profile=profile,
        runs=runs,
        root=root,
        seed=seed,
        authorized=authorized,
        yes=yes,
        max_requests=None,
        resume=None,
    )


def _finish(result: SweepResult) -> None:
    render_sweep(result, console)
    if result.store is not None:
        console.print(f"\n[dim]artifacts in {result.store.run_dir}[/dim]")
    if result.status is SweepStatus.REFUSED:
        raise typer.Exit(code=2)
    if result.status is SweepStatus.DECLINED:
        raise typer.Exit(code=0)
