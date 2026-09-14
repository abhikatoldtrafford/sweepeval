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
from sweepeval.judge.client import JudgeConfig
from sweepeval.pipeline import rank_sweep
from sweepeval.rank.constraints import DEFAULT_CONSTRAINTS, Constraint
from sweepeval.report.frontier import render_frontier, render_preference
from sweepeval.report.frontier_json import frontier_payload
from sweepeval.report.sweep import render_sweep
from sweepeval.store.derived import write_derived

console = Console()


def _confirmer(
    yes: bool, *, no_input: bool = False, cap: BudgetCap | None = None
) -> object:
    """Build the pre-flight predicate (§12.3)."""

    def confirm(estimate: Estimate) -> bool:
        console.print("\n[bold]before spending anything[/bold]")
        console.print(render_estimate(estimate))
        if cap is not None and cap.exceeded_by(estimate):
            # The cap will bind mid-sweep rather than up front, so say so now
            # instead of letting INCOMPLETE arrive as a surprise at the end.
            console.print(
                f"\n[yellow]this estimate exceeds your {cap.unit} cap of "
                f"{cap.value}[/yellow] — the sweep will stop between configs "
                "when the cap is reached and report INCOMPLETE, naming every "
                "config that never ran."
            )
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


def _judge(
    model: str | None, url: str | None, key: str | None, target_key: str | None
) -> JudgeConfig | None:
    """Build the judge config, or refuse clearly.

    A model with no endpoint is the likeliest mistake, and defaulting it to the
    target's URL would silently produce a model scoring its own output -- which
    §11.9 refuses anyway, but at the cost of a confusing error much later.
    """
    if model is None:
        if url or key:
            raise typer.BadParameter("--judge-url/--judge-key need --judge")
        return None
    if not url:
        raise typer.BadParameter(
            "--judge needs --judge-url. The judge may not be the target (§11.9), "
            "so there is no safe default."
        )
    return JudgeConfig(model=model, url=url, key=key or target_key)


def _cap(
    max_requests: int | None,
    max_tokens: int | None = None,
    max_dollars: float | None = None,
) -> BudgetCap | None:
    """The one cap the run is held to.

    All three units bind in the engine, and for a while only requests was
    reachable from the CLI -- so the token and dollar caps were library-API
    only, and the bugs in them went unnoticed because nothing exercised them.

    One at a time on purpose: two caps in different units raise the question
    of which one stopped the run, and the answer belongs in the stop reason,
    not in the user's head.
    """
    given = [
        (max_requests, "requests"),
        (max_tokens, "tokens"),
        (max_dollars, "dollars"),
    ]
    chosen = [(v, unit) for v, unit in given if v]
    if not chosen:
        return None
    if len(chosen) > 1:
        names = ", ".join(f"--max-{unit}" for _v, unit in chosen)
        raise typer.BadParameter(f"give one cap, not {len(chosen)}: {names}")
    value, unit = chosen[0]
    return BudgetCap(value=value, unit=unit)  # type: ignore[arg-type]


def sweep_command(
    url: str | None = typer.Argument(
        None, help="The endpoint to sweep. Optional when --config supplies one."
    ),
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
    max_tokens: int | None = typer.Option(
        None, "--max-tokens", help="Hard cap in tokens; stops between configs."
    ),
    max_dollars: float | None = typer.Option(
        None,
        "--max-dollars",
        help="Hard cap in currency. Needs pricing in --config; inert without it.",
    ),
    judge: str | None = typer.Option(
        None,
        "--judge",
        help=(
            "Model id for §11.9 judge escalation. Resolves the ambiguities a "
            "deterministic contract declared it could not settle. Off by default; "
            "it spends, and its worst case is in the pre-flight."
        ),
    ),
    judge_url: str | None = typer.Option(
        None, "--judge-url", help="Judge endpoint. Must not be the target."
    ),
    judge_key: str | None = typer.Option(
        None, "--judge-key", help="Judge credential. Defaults to --key."
    ),
    resume: str | None = typer.Option(
        None, "--resume", help="Continue an existing run id."
    ),
    objectives: str | None = typer.Option(
        None, "--objectives", help="Narrow the frontier, e.g. security,latency."
    ),
    prefer: str | None = typer.Option(
        None,
        "--prefer",
        help=(
            "Apply your own priority to the frontier: 'security,cost,latency' "
            "or 'maximize security_pass_rate subject to latency_p95_ms < 2000'."
        ),
    ),
    alpha: float = typer.Option(0.05, "--alpha", help="Family-wise error rate."),
    fmt: str | None = typer.Option(None, "--format", help="html,junit,json"),
    max_configs: int | None = typer.Option(
        None, "--max-configs",
        help="Override the profile's config cap, up or down. The estimate follows it.",
    ),
    config: Path | None = typer.Option(
        None, "--config", "-c",
        help="A sweepeval.yaml: corrections, declared axes, pricing, constraints.",
    ),
) -> None:
    """Discover, plan and sweep every discoverable configuration."""
    declared = None
    if config is not None:
        from sweepeval.execute.declared import load_declared

        declared = load_declared(config)
        for warning in declared.warnings:
            console.print(f"[yellow]{config}: {warning}[/yellow]")
        url = url or declared.url
        profile = declared.profile or profile
        runs = declared.runs or runs
        objectives = objectives or (
            ",".join(declared.objectives) if declared.objectives else None
        )
        if declared.describe():
            console.print("[bold]from your config[/bold]")
            for line in declared.describe():
                console.print(f"  {line}")

    if not url:
        raise typer.BadParameter(
            "give an endpoint URL, or a --config whose target.url names one"
        )

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
                yes, no_input=False, cap=_cap(max_requests, max_tokens, max_dollars)
            ),
            cap=_cap(max_requests, max_tokens, max_dollars),
            judge=_judge(judge, judge_url, judge_key, key),
            resume_run_id=resume,
            config_cap=max_configs,
            declared=declared,
            pricing=declared.pricing if declared else None,
            on_progress=lambda msg: console.print(f"[dim]{msg}[/dim]"),
        )
    )
    _finish(
        result, objectives=objectives, prefer=prefer, alpha=alpha, seed=seed, fmt=fmt,
        constraints=declared.constraints if declared and declared.constraints else None,
    )


def run_command(
    url: str = typer.Argument(..., help="The endpoint to evaluate."),
    key: str | None = typer.Option(None, "--key", "-k"),
    profile: str = typer.Option("quick", "--profile", help="quick | standard | deep"),
    runs: int = typer.Option(3, "-n", "--runs"),
    root: Path = typer.Option(Path(".sweepeval"), "--root"),
    seed: int = typer.Option(0, "--seed"),
    authorized: bool = typer.Option(False, "--i-am-authorized"),
    yes: bool = typer.Option(False, "--yes", "-y"),
    prefer: str | None = typer.Option(None, "--prefer"),
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
        max_tokens=None,
        max_dollars=None,
        judge=None,
        judge_url=None,
        judge_key=None,
        resume=None,
        max_configs=None,
        objectives=None,
        prefer=prefer,
        alpha=0.05,
        fmt=None,
        config=None,
    )


def _finish(
    result: SweepResult,
    *,
    objectives: str | None = None,
    prefer: str | None = None,
    alpha: float = 0.05,
    seed: int = 0,
    fmt: str | None = None,
    constraints: tuple[Constraint, ...] | None = None,
) -> None:
    render_sweep(result, console)

    frontier = None
    if result.configs:
        frontier = _rank(
            result, objectives=objectives, prefer=prefer, alpha=alpha, seed=seed,
            constraints=constraints,
        )
    if result.store is not None:
        from sweepeval.report.stored import write_aggregates

        write_aggregates(result)
    _emit(result, frontier, fmt)

    if result.store is not None:
        console.print(f"\n[dim]artifacts in {result.store.run_dir}[/dim]")
    if result.status is SweepStatus.REFUSED:
        raise typer.Exit(code=2)
    if result.status is SweepStatus.DECLINED:
        raise typer.Exit(code=0)


def _rank(
    result: SweepResult,
    *,
    objectives: str | None,
    prefer: str | None,
    alpha: float,
    seed: int,
    constraints: tuple[Constraint, ...] | None = None,
):
    """Rank, report and store the frontier (§14, §6.2)."""
    names = [n.strip() for n in (objectives or "").split(",") if n.strip()]
    try:
        frontier = rank_sweep(
            result, objectives=names, alpha=alpha, seed=seed,
            constraints=constraints or DEFAULT_CONSTRAINTS,
        )
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error

    labels = {row.config_id: row.config.label() for row in result.configs}
    render_frontier(frontier, labels, console)

    if result.store is not None:
        write_derived(
            result.store.frontier_path,
            frontier_payload(frontier, labels),
            derived_from=result.store.provenance(),
        )

    if not prefer:
        return frontier

    from sweepeval.rank.prefer import apply_preference, parse_preference

    try:
        preference = parse_preference(prefer)
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error

    outcome = apply_preference(
        preference, frontier, {r.config_id: r.metrics for r in result.configs}
    )
    render_preference(outcome, labels, console)
    return frontier


def _emit(result: SweepResult, frontier, fmt: str | None) -> None:
    """Machine formats only when asked (§15)."""
    if not fmt or result.store is None:
        return

    from sweepeval.report.html import as_html, as_junit

    wanted = {f.strip() for f in fmt.split(",") if f.strip()}
    written: list[str] = []
    for name in sorted(wanted):
        if name == "html":
            result.store.report_path("html").write_text(
                as_html(result, frontier), encoding="utf-8"
            )
            written.append("report.html")
        elif name == "junit":
            result.store.report_path("xml").write_text(
                as_junit(result, frontier), encoding="utf-8"
            )
            written.append("report.xml")
        elif name == "json":
            written.append("frontier.json")
        else:
            console.print(f"[yellow]unknown --format {name!r}, skipped[/yellow]")
    if written:
        console.print(
            f"\nwrote {', '.join(written)} to {result.store.run_dir}"
        )
