"""``sweepeval rejudge`` (spec §11.9, §5.1).

Escalates a stored run's ambiguities to an LLM judge without re-running the
target. The judge decides from the response text and the response text is
already in the store, so paying for the whole run again to use it is the wrong
price.

Writes a **new** run. The source is append-only and is what was paid for (I7),
and `judge` is a hard comparability key -- a judged result and an unjudged one
are different measurements, which is why they must be two runs that `compare`
refuses to put side by side.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from sweepeval.execute.rejudge import rejudge_run
from sweepeval.judge import JudgeConfig, JudgeError
from sweepeval.stats.diff import ExitCode

console = Console()


def rejudge_command(
    run_dir: Path = typer.Argument(..., help="A stored run directory."),
    judge: str = typer.Option(
        ..., "--judge",
        help="Judge model id. Must not be a model the run measured.",
    ),
    judge_url: str = typer.Option(..., "--judge-url", help="Judge endpoint."),
    judge_key: str | None = typer.Option(None, "--judge-key"),
    root: Path = typer.Option(
        Path(".sweepeval"), "--root", help="Where the judged run is written."
    ),
    seed: int = typer.Option(0, "--seed"),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Spend the judge calls without prompting."
    ),
) -> None:
    """Judge a stored run's ambiguities. Spends judge calls, not target calls."""
    from sweepeval.execute.rescore import rescore_run

    # I9: the estimate comes before the spend, and here it is exact rather
    # than estimated -- the ambiguities are already on disk and counting them
    # sends nothing.
    try:
        preview = rescore_run(run_dir, seed=seed)
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error

    if not preview.escalatable:
        console.print(
            "[green]nothing to judge:[/green] no observation in this run is "
            "marked ambiguous with a stored response to show a judge."
        )
        raise typer.Exit(code=0)

    console.print(
        f"{preview.escalatable} ambiguity(s) to judge, one call each, "
        f"against [bold]{judge}[/bold]. No target request will be made."
    )
    if preview.unjudgeable:
        # Said out loud, every time. An ambiguity nobody resolved and nobody
        # mentioned is the silent gap this feature exists to close, and
        # quietly subtracting it from the count reopens it one row at a time.
        console.print(
            f"[yellow]{preview.unjudgeable} further ambiguity(s) have no "
            f"stored response and cannot be judged at any price; they keep "
            f"their UNSCORABLE[/yellow]"
        )
    if not yes and not typer.confirm("spend them?", default=False):
        console.print("[yellow]nothing was sent[/yellow]")
        raise typer.Exit(code=0)

    try:
        result = rejudge_run(
            run_dir,
            judge=JudgeConfig(model=judge, url=judge_url, key=judge_key),
            root=root, seed=seed,
        )
    except JudgeError as error:
        # §11.9's independence refusal. A usage error: the flag is wrong for
        # this run, which is knowable from the command line and the plan.
        console.print(f"[red]REFUSED[/red]    {error}")
        raise typer.Exit(code=int(ExitCode.USAGE_ERROR)) from None
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error

    if result.warning:
        console.print(f"[yellow]{result.warning}[/yellow]")

    console.print(
        f"\njudged {result.resolved} of {result.escalations} ambiguity(s) in "
        f"{result.judge_requests} request(s)"
    )
    if result.failures:
        # A judge that fails is not a judge that passed.
        console.print(
            f"[yellow]{len(result.failures)} call(s) errored or returned "
            f"unusable JSON; those rows keep their deterministic "
            f"UNSCORABLE[/yellow]"
        )
        for unit_id, reason in result.failures[:5]:
            console.print(f"  {unit_id.split('#')[0]}: {reason}")

    if result.skipped:
        console.print(
            f"[yellow]{len(result.skipped)} ambiguity(s) the judge could not "
            f"be asked about[/yellow]"
        )
        for unit_id, reason in result.skipped[:5]:
            console.print(f"  {unit_id.split('#')[0]}: {reason}")

    _render_coverage(result)
    console.print(f"\nwrote the judged run to {result.run_dir}")
    console.print(
        "[dim]the source run is unchanged. `compare` will refuse to put the "
        "two side by side: a judged result and an unjudged one are different "
        "measurements (§6.5).[/dim]"
    )


def _render_coverage(result: object) -> None:
    """Coverage before and after, per config. The finding, in one table.

    Coverage rather than the rate, because the rate is not the claim: the
    judge exists to shrink the band nothing could score, and a pass rate that
    moved while coverage did not would mean something else entirely.
    """
    from rich.table import Table

    before = result.coverage_before  # type: ignore[attr-defined]
    after = result.after  # type: ignore[attr-defined]
    coverage_after = result.coverage_after  # type: ignore[attr-defined]

    table = Table(title="guardrail coverage", header_style="bold")
    table.add_column("config")
    table.add_column("scored before", justify="right")
    table.add_column("scored after", justify="right")
    table.add_column("of", justify="right")
    table.add_column("pass rate after", justify="right")

    for config_id in sorted(before):
        was = before[config_id].get("guardrail")
        now = coverage_after.get(config_id, {}).get("guardrail")
        if not was or not now:
            continue
        value = after.get(config_id, {}).get("guardrail_pass_rate")
        rate = "—"
        if value is not None and value.point is not None:
            rate = f"{value.point:.3f}"
            if value.lo is not None and value.hi is not None:
                rate += f" [{value.lo:.3f}, {value.hi:.3f}]"
        table.add_row(config_id, str(was[0]), str(now[0]), str(now[1]), rate)

    console.print()
    console.print(table)
