"""Terminal report for a sweep (spec §15, §12.5, D15).

Deliberately does **not** rank. The layering contract keeps domination in one
module, and this reporter is not it — what is printed here is every config's
measurements side by side, with intervals, and nothing that implies an order.
The frontier is added by M9, through the ranker.

Two things this must never do quietly:

* Present an ``INCOMPLETE`` sweep as a sweep. Configs the budget cap stopped
  before are named, because a table missing three rows looks exactly like a
  table of nine configs.
* Print a single-config sweep as though a search had happened. D15's banner
  lists every candidate axis and why it was rejected, which for a custom agent
  system is the modal outcome rather than an edge case.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sweepeval.report.terminal import format_flags, format_interval, format_point
from sweepeval.schema.metric import Flag, MetricValue

__all__ = ["render_sweep"]

_HEADLINE = (
    "security_pass_rate",
    "guardrail_pass_rate",
    "context_retention_auc",
    "target_determinism_at_temp0",
    "latency_ms",
    "error_rate",
)


def render_sweep(result: Any, console: Console | None = None) -> None:
    """Render a completed (or stopped) sweep."""
    console = console or Console()

    status = result.status.value
    style = "green" if status == "COMPLETE" else "yellow"
    console.print(
        Panel(
            f"[bold]{'single-configuration evaluation' if result.single_config else 'sweep'}"
            f"[/bold]  [{style}]{status}[/{style}]\n"
            f"target      {_target(result)}\n"
            f"profile     {result.profile}  ({result.corpus.unit_count} units, "
            f"{result.corpus.calls_per_run} calls/run, {result.runs} runs)\n"
            f"configs     {len(result.configs)} executed"
            f"{f' of {len(result.plan.configs)} planned' if result.not_run else ''}\n"
            f"run         {result.run_id}",
            title="sweepeval",
            expand=False,
        )
    )

    if result.stop_reason:
        console.print(f"[{style}]{result.stop_reason}[/{style}]\n")

    if result.status.value in {"DECLINED", "REFUSED"}:
        return

    _profile_note(console, result)
    _axes(console, result)
    _matrix(console, result)
    _coverage(console, result)
    _cache(console, result)
    _cost(console, result)
    _not_run(console, result)
    _skipped(console, result)
    _assumptions(console, result)


def _profile_note(console: Console, result: Any) -> None:
    """§10.2, §15: `quick` is fast because it is smaller, and it says so.

    The wording matters. An earlier 16-unit `quick` genuinely produced no
    valid intervals; the 40-unit one produces valid intervals that are wide.
    Writing the first when the second is true tells the user their numbers are
    unusable when they are merely imprecise.
    """
    if result.profile != "quick":
        return
    console.print(
        "[yellow]profile quick[/yellow]: intervals are valid but wide, so few "
        "pairs will separate.\nThese results are [bold]not gate-eligible[/bold] "
        "— use --profile standard for decisions or gating.\n"
    )


def _coverage(console: Console, result: Any) -> None:
    """§14.5, §11.8: silently-excluded trials have to be visible."""
    rows = [(row.config_id, row.coverage) for row in result.configs]
    interesting = [
        (config_id, coverage)
        for config_id, coverage in rows
        if any(s < a for s, a in coverage.values())
    ]
    if not interesting:
        return

    console.print("\n[bold]coverage[/bold] (scored / attempted)")
    for config_id, coverage in interesting:
        parts = [
            f"{family} {scored}/{attempted}"
            for family, (scored, attempted) in sorted(coverage.items())
            if attempted
        ]
        console.print(f"  {config_id:10s} {'  '.join(parts)}")


def _target(result: Any) -> str:
    discovery = result.discovery
    if discovery is None:
        return "(not discovered)"
    return str(discovery.payload["target"]["url"])


def _axes(console: Console, result: Any) -> None:
    """D15 and D22: what is being swept, and what was rejected or shrunk."""
    plan = result.plan

    if plan.axes:
        console.print("[bold]axes[/bold]")
        for axis in sorted(plan.axes):
            values = ", ".join(str(v) for v in plan.axes[axis])
            console.print(f"  {axis:16s} {values}")
    else:
        console.print(
            "[bold]no variable axis survived discovery[/bold] — this is a single "
            "configuration,\nnot a sweep."
        )

    if plan.rejected_axes:
        console.print("\n[bold]rejected axes[/bold]")
        for axis, reason in plan.rejected_axes:
            console.print(f"  {axis:16s} {reason}")

    if plan.shrink_steps:
        console.print(f"\n[bold]shrink ladder[/bold] (cap {plan.cap})")
        for step in plan.shrink_steps:
            console.print(f"  {step}")

    if plan.dropped_models:
        console.print("\n[bold]models dropped[/bold]")
        for model_id, reason in plan.dropped_models[:10]:
            console.print(f"  {model_id:24s} {reason}")
        if len(plan.dropped_models) > 10:
            console.print(f"  ... and {len(plan.dropped_models) - 10} more")

    console.print()


def _matrix(console: Console, result: Any) -> None:
    """Every config's measurements, with intervals. No ordering implied."""
    if not result.configs:
        console.print("[red]no configuration produced a measurement[/red]")
        return

    present = [
        m for m in _HEADLINE if any(m in row.metrics for row in result.configs)
    ]
    if not present:
        console.print("[red]no metric produced a value in any configuration[/red]")
        return

    # The config id is the column, and the labels are a legend above it. Six
    # metric columns of "point [lo, hi] FLAGS" leave a label column narrow
    # enough that rich truncates every row to the same "tempera… sys=none",
    # which is worse than useless: the table stops distinguishing the configs
    # it exists to compare.
    console.print("[bold]configurations[/bold]")
    for row in result.configs:
        console.print(f"  {row.config_id:8s} {row.config.label()}")
    console.print()

    table = Table(
        title="measurements (unranked — the frontier is computed by `rank`)",
        header_style="bold",
    )
    table.add_column("config", no_wrap=True)
    for metric in present:
        table.add_column(_short(metric), justify="right")

    # INDICATIVE is a property of the profile, not of an individual number.
    # Repeating it in all forty-two cells of a quick run buries the flags that
    # do vary — LOW_N, LOW_COVERAGE, CACHE_SUSPECTED — in noise, so it is
    # stated once below the table instead.
    hide = frozenset({Flag.INDICATIVE}) if result.profile == "quick" else frozenset()

    for row in result.configs:
        cells = [row.config_id]
        for metric in present:
            value = row.metrics.get(metric)
            cells.append(_cell(value, hide=hide))
        table.add_row(*cells)
    console.print(table)

    if any(
        Flag.INDICATIVE in v.flags
        for row in result.configs
        for v in row.metrics.values()
    ):
        console.print(
            "[yellow]INDICATIVE[/yellow]: intervals are valid but wide, so few "
            "pairs will separate.\nUse --profile standard for decisions or gating."
        )


def _short(metric: str) -> str:
    """Header names that survive six columns of intervals side by side."""
    return {
        "security_pass_rate": "security",
        "guardrail_pass_rate": "guardrail",
        "context_retention_auc": "retention",
        "target_determinism_at_temp0": "det@temp0",
        "config_repeatability": "repeatable",
        "latency_ms": "latency ms",
        "error_rate": "error rate",
        "tokens_out": "tokens out",
    }.get(metric, metric.replace("_", " "))


def _cell(value: MetricValue | None, *, hide: frozenset[Flag] = frozenset()) -> str:
    if value is None:
        return "[dim]—[/dim]"
    shown = value.model_copy(update={"flags": tuple(f for f in value.flags if f not in hide)})
    flags = format_flags(shown)
    return f"{format_point(value)} {format_interval(value)}" + (
        f" {flags}" if flags else ""
    )


def _cache(console: Console, result: Any) -> None:
    """§12.7: a suspected cache invalidates determinism and latency."""
    suspected = [row for row in result.configs if row.cache.suspected]
    if not suspected:
        return
    console.print("\n[red]CACHE_SUSPECTED[/red]")
    for row in suspected:
        console.print(f"  {row.config_id}: {row.cache.reason}")
        for evidence in row.cache.evidence[:2]:
            console.print(f"    {evidence.describe()}")


def _cost(console: Console, result: Any) -> None:
    """§12.4: say which number this is and where it came from."""
    rows = [row for row in result.configs if row.cost is not None]
    if not rows:
        return
    console.print("\n[bold]cost[/bold]")
    for row in rows:
        assert row.cost is not None
        console.print(f"  {row.config_id:10s} {row.cost.describe()}")


def _not_run(console: Console, result: Any) -> None:
    if not result.not_run:
        return
    console.print(
        f"\n[yellow]{len(result.not_run)} configuration(s) never ran[/yellow] — "
        "they are absent from the table above, not tied:"
    )
    for config_id in result.not_run:
        planned = next(
            (c for c in result.plan.configs if c.config_id == config_id), None
        )
        console.print(f"  {config_id:10s} {planned.label() if planned else ''}")


def _skipped(console: Console, result: Any) -> None:
    """I5: never a silent absence."""
    if not result.skipped:
        return
    console.print("\n[bold]SKIPPED[/bold]")
    for family, reason in sorted(result.skipped):
        console.print(f"  {family:16s} {reason}")
    if result.families_not_run:
        console.print(
            f"  probes for {', '.join(result.families_not_run)} were not executed, "
            "so their calls were not spent"
        )


def _assumptions(console: Console, result: Any) -> None:
    """§8.6: discovery is a bootstrap the user can correct."""
    if not result.assumptions:
        return
    console.print("\n[yellow]assumptions you can correct[/yellow]")
    for field_name, confidence, detail in result.assumptions:
        console.print(f"  {field_name} ({confidence}) — {detail}")
