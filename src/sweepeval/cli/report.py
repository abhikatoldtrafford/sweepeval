"""``sweepeval report`` (spec §15, D33).

Re-renders a stored run. Sends nothing, needs no credentials, and works with
the network off — which is what makes a committed example run usable as
evidence, and what makes "changing your preference needs no re-run" true.

Re-ranking happens here too, from the stored cluster tables. That is why they
are stored: without them ``--prefer`` and ``--objectives`` would quietly
require the target again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from sweepeval.report.frontier import render_frontier, render_preference
from sweepeval.report.frontier_json import frontier_payload
from sweepeval.report.html import as_html, as_junit
from sweepeval.report.stored import StoredRun, load_run
from sweepeval.report.sweep import render_sweep
from sweepeval.store.derived import provenance_of, write_derived

console = Console()


def report_command(
    run_dir: Path = typer.Argument(..., help="A stored run directory."),
    fmt: str = typer.Option(
        "terminal", "--format", help="terminal,html,junit,json,md"
    ),
    objectives: str | None = typer.Option(None, "--objectives"),
    prefer: str | None = typer.Option(None, "--prefer"),
    alpha: float = typer.Option(0.05, "--alpha"),
    seed: int = typer.Option(0, "--seed"),
    out: Path | None = typer.Option(
        None, "--out", help="Write the file here instead of into the run directory."
    ),
) -> None:
    """Rebuild a report from a stored run, offline."""
    try:
        run = load_run(run_dir)
    except FileNotFoundError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error

    if run.stale:
        # I7: a derived file is regenerable, so the answer is to regenerate,
        # not to refuse. But reporting numbers off an aggregates.json whose
        # logs have since grown, without saying so, is how a stale frontier
        # gets quoted as a current one.
        console.print(
            "[yellow]aggregates.json is stale: the logs beside it have "
            "changed since it was written, so these numbers are not the "
            "whole run. Re-rank from the logs to refresh it.[/yellow]"
        )

    frontier = _rank(run, objectives, alpha, seed)
    labels = {row.config_id: row.config.label() for row in run.configs}

    for name in [f.strip() for f in fmt.split(",") if f.strip()]:
        if name == "terminal":
            render_sweep(run, console)
            if frontier is not None:
                render_frontier(frontier, labels, console)
                _prefer(frontier, run, prefer, labels)
        elif name == "html":
            _write(run, out, "report.html", as_html(run, frontier))
        elif name == "junit":
            _write(run, out, "report.xml", as_junit(run, frontier))
        elif name == "json":
            if frontier is None:
                console.print("[yellow]nothing to rank, so no frontier.json[/yellow]")
            else:
                _write_json(
                    run, out, "frontier.json", frontier_payload(frontier, labels)
                )
        elif name == "md":
            _write(run, out, "report.md", markdown(run, frontier))
        else:
            console.print(f"[yellow]unknown --format {name!r}, skipped[/yellow]")


def _rank(run: StoredRun, objectives: str | None, alpha: float, seed: int) -> Any:
    if not run.configs:
        return None
    from sweepeval.pipeline import rank_sweep

    names = [n.strip() for n in (objectives or "").split(",") if n.strip()]
    try:
        return rank_sweep(run, objectives=names, alpha=alpha, seed=seed)
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error


def _prefer(
    frontier: Any, run: StoredRun, prefer: str | None, labels: dict[str, str]
) -> None:
    if not prefer:
        return
    from sweepeval.rank.prefer import apply_preference, parse_preference

    try:
        preference = parse_preference(prefer)
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error

    outcome = apply_preference(
        preference, frontier, {r.config_id: r.metrics for r in run.configs}
    )
    render_preference(outcome, labels, console)


def _destination(run: StoredRun, out: Path | None, name: str) -> Path:
    if out is None:
        return (run.run_dir or Path(".")) / name
    return out if out.suffix else out / name


def _write(run: StoredRun, out: Path | None, name: str, text: str) -> None:
    path = _destination(run, out, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    console.print(f"wrote {path}")


def _write_json(run: StoredRun, out: Path | None, name: str, payload: Any) -> None:
    """A re-derived frontier records the logs it was re-derived from (I7).

    Those are the stored run's logs, not the destination's: `--out` can put
    the file anywhere, and provenance that pointed at wherever it landed would
    name nothing.
    """
    path = _destination(run, out, name)
    write_derived(path, payload, derived_from=provenance_of(run.run_dir))
    console.print(f"wrote {path}")


def markdown(run: StoredRun, frontier: Any = None) -> str:
    """A summary for a PR comment or a CI job summary."""
    from sweepeval.schema.metric import Flag

    lines = [
        f"# sweepeval - {run.profile} sweep",
        "",
        f"{run.run_id} - {len(run.configs)} configuration(s) - {run.runs} runs - "
        f"**{run.status.value}**",
        "",
    ]
    if run.stop_reason:
        lines += [f"> {run.stop_reason}", ""]
    if run.profile == "quick":
        lines += [
            "> Intervals at `quick` are valid but wide, so few pairs will "
            "separate. **Not gate-eligible** - use `--profile standard` for "
            "decisions or gating.",
            "",
        ]

    names: list[str] = []
    for row in run.configs:
        for metric in row.metrics:
            if metric not in names:
                names.append(metric)
    names.sort()

    if names:
        lines += [
            "| config | " + " | ".join(names) + " |",
            "|" + "---|" * (len(names) + 1),
        ]
        for row in run.configs:
            cells = []
            for name in names:
                value = row.metrics.get(name)
                if value is None or Flag.NO_VALID_INTERVAL in value.flags:
                    cells.append("n/a")
                else:
                    cells.append(f"{value.point:.4g} [{value.lo:.3g}, {value.hi:.3g}]")
            lines.append(f"| {row.config.label()} | " + " | ".join(cells) + " |")
        lines.append("")

    if frontier is not None:
        lines += [
            "## frontier",
            "",
            f"{len(frontier.frontier)} non-dominated configuration(s) in "
            f"{len(frontier.clusters)} tied cluster(s), alpha {frontier.alpha:g} "
            "family-wise.",
            "",
        ]
        for index, cluster in enumerate(frontier.clusters, start=1):
            lines.append(f"- **cluster {index}**: {', '.join(cluster.members)}")
        if frontier.dominated:
            lines.append("")
            for config_id, dominators in sorted(frontier.dominated.items()):
                lines.append(f"- {config_id} dominated by {', '.join(dominators)}")
        lines.append("")

    if run.skipped:
        lines += ["## SKIPPED", ""]
        lines += [f"- **{f}** - {r}" for f, r in sorted(run.skipped)]
        lines.append("")

    return "\n".join(lines)
