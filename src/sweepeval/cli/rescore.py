"""``sweepeval rescore`` (spec §5.1, I7).

Recomputes a stored run's verdicts with this build's scorers, from the
response text the store kept. Sends nothing and needs no credentials.

Prints what *would* change and never rewrites `observations.jsonl`: that log
is what was paid for, and it is append-only.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from sweepeval.execute.rescore import RESCORABLE, rescore_run

console = Console()


def rescore_command(
    run_dir: Path = typer.Argument(..., help="A stored run directory."),
    families: str = typer.Option(
        ",".join(RESCORABLE), "--families", help=",".join(RESCORABLE)
    ),
    seed: int = typer.Option(0, "--seed"),
    metrics: bool = typer.Option(
        False, "--metrics", help="Also print the re-aggregated metrics."
    ),
) -> None:
    """Re-score a stored run offline and report what changed."""
    selected = tuple(f.strip() for f in families.split(",") if f.strip())
    try:
        report = rescore_run(run_dir, families=selected, seed=seed)
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error

    console.print(
        f"re-scored {report.considered} verdict(s) across "
        f"{', '.join(report.families)}"
    )
    if report.unjoinable:
        # Loud, because a run whose text is gone cannot be re-scored and
        # "no changes" would otherwise read as agreement.
        console.print(
            f"[yellow]{report.unjoinable} row(s) had no stored text and could "
            f"not be re-scored[/yellow]"
        )

    if not report.changes:
        console.print("[green]no verdict changed[/green]")
    for change in report.changes:
        console.print(f"  {change}")

    if report.mismatched:
        # The corrected numbers are only trustworthy if the offline path
        # reproduces the run on the configs it did not touch.
        console.print(
            "\n[red]offline aggregation does not reproduce the stored run:[/red]"
        )
        for line in report.mismatched[:10]:
            console.print(f"  {line}")
        raise typer.Exit(code=2)

    console.print(
        f"\n[dim]reproduced {report.reproduced} stored metric(s) exactly on "
        f"{len(report.metrics) - len(report.touched)} untouched config(s)[/dim]"
    )

    if metrics:
        for config_id in sorted(report.metrics):
            console.print(f"\n{config_id}")
            for name, value in sorted(report.metrics[config_id].items()):
                if value.point is None:
                    continue
                interval = (
                    f"[{value.lo:.3f}, {value.hi:.3f}]"
                    if value.lo is not None and value.hi is not None
                    else "(no interval)"
                )
                console.print(f"  {name:32} {value.point:10.3f} {interval}")
