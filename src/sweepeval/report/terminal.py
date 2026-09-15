"""Terminal report (spec §15).

Printed on every run. What it must always show, because a number without these
is not interpretable:

* every metric with its interval and its flags;
* every ``SKIPPED`` family and the reason (I5);
* every low-confidence inference, as an assumption the user can correct;
* coverage, so silently-excluded trials are visible;
* the banner when this was a single-configuration evaluation rather than a
  sweep, with each rejected axis and why (D15).
"""

from __future__ import annotations

from collections.abc import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sweepeval.schema.metric import CIMethod, Flag, MetricValue

__all__ = ["format_flags", "format_interval", "format_point", "render_evaluation"]

_FLAG_STYLE = {
    Flag.INDICATIVE: "yellow",
    Flag.LOW_N: "yellow",
    Flag.LOW_COVERAGE: "red",
    Flag.NO_VALID_INTERVAL: "red",
    Flag.CACHE_SUSPECTED: "red",
}


def format_interval(value: MetricValue) -> str:
    if value.method is CIMethod.none or value.lo is None or value.hi is None:
        return "[red]no valid interval[/red]"
    return f"[{value.lo:.3g}, {value.hi:.3g}]"


def format_point(value: MetricValue) -> str:
    """The number, or an em dash when there is no number to show.

    A metric with NO_VALID_INTERVAL carries a placeholder point. Printing it
    reads as a measurement — "guardrail_pass_rate 0.000" says the target failed
    every guardrail, when what happened is that nothing could be scored.
    """
    if Flag.NO_VALID_INTERVAL in value.flags:
        return "[dim]—[/dim]"
    return f"{value.point:.4g}"


def format_flags(value: MetricValue) -> str:
    if not value.flags:
        return ""
    return " ".join(
        f"[{_FLAG_STYLE.get(f, 'yellow')}]{f.value}[/]" for f in value.flags
    )


def render_evaluation(result: object, console: Console | None = None) -> None:
    """Render a single-configuration evaluation."""
    console = console or Console()

    # The model belongs on the header for the same reason the shape does: it
    # is what was measured, and a reader of a baseline or a gate has no other
    # way to see which one answered.
    model = getattr(result, "model", None)
    console.print(
        Panel(
            f"[bold]single-configuration evaluation[/bold]\n"
            f"target      {result.discovery.payload['target']['url']}\n"  # type: ignore[attr-defined]
            f"shape       {result.discovery.ladder.shape.name}\n"  # type: ignore[attr-defined]
            f"type        {result.discovery.target_type}\n"  # type: ignore[attr-defined]
            + (f"model       {model}\n" if model else "")
            + f"profile     {result.profile}  ({result.corpus.unit_count} units, "  # type: ignore[attr-defined]
            f"{result.corpus.calls_per_run} calls/run)\n"  # type: ignore[attr-defined]
            f"run         {result.run_id}",  # type: ignore[attr-defined]
            title="sweepeval",
            expand=False,
        )
    )

    _metrics_table(console, result)
    _retention(console, result)
    _axes(console, result)
    _skipped(console, result)
    _coverage(console, result)
    _assumptions(console, result)


def _metrics_table(console: Console, result: object) -> None:
    metrics: dict[str, MetricValue] = result.metrics  # type: ignore[attr-defined]
    if not metrics:
        console.print("[red]no metric produced a value[/red]")
        return

    table = Table(title="metrics", header_style="bold")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_column("95% interval", justify="right")
    table.add_column("clusters", justify="right")
    table.add_column("flags")

    for name in sorted(metrics):
        value = metrics[name]
        table.add_row(
            name,
            format_point(value),
            format_interval(value),
            str(value.n_clusters),
            format_flags(value),
        )
    console.print(table)

    if any(Flag.INDICATIVE in v.flags for v in metrics.values()):
        console.print(
            "[yellow]INDICATIVE[/yellow]: intervals are valid but wide, so few "
            "pairs will separate. Use --profile standard for decisions or gating."
        )


def _retention(console: Console, result: object) -> None:
    """The decay curve and the depth where retention crosses the floor (§11.5)."""
    curve: dict[int, float] = getattr(result, "retention_curve", {})
    if not curve:
        return

    weights: dict[int, float] = getattr(result, "retention_weights", {})
    console.print("\n[bold]context retention by depth[/bold]")
    for depth in sorted(curve):
        bar = "#" * round(curve[depth] * 20)
        share = f"  (AUC weight {weights[depth]:.0%})" if depth in weights else ""
        console.print(f"  depth {depth:>3}  {curve[depth]:>5.2f}  {bar}{share}")

    floor = getattr(result, "retention_depth_at_floor", None)
    if floor is not None:
        console.print(f"  retention crosses 0.5 at depth {floor}")

    if len(curve) == 1:
        console.print(
            "  [yellow]one measured depth: the AUC is that depth's recall, not "
            "an area[/yellow]"
        )


def _axes(console: Console, result: object) -> None:
    """D15: a frontier of one is reported as a frontier of one."""
    rejected: Sequence[tuple[str, str]] = result.axes_rejected  # type: ignore[attr-defined]
    if not rejected:
        return
    console.print("\n[bold]no variable axes discovered[/bold] — this is a single")
    console.print("configuration, not a sweep. Candidate axes and why each was rejected:")
    for axis, reason in rejected:
        console.print(f"  {axis:16s} {reason}")


def _skipped(console: Console, result: object) -> None:
    """I5: never a silent absence."""
    skipped: Sequence[tuple[str, str]] = result.skipped  # type: ignore[attr-defined]
    if not skipped:
        return
    console.print("\n[bold]SKIPPED[/bold]")
    for family, reason in sorted(skipped):
        console.print(f"  {family:16s} {reason}")


def _coverage(console: Console, result: object) -> None:
    from sweepeval.stats.aggregate import coverage_of

    observations = result.observations  # type: ignore[attr-defined]
    rows = []
    for family in ("security", "guardrail", "operational"):
        for _, coverage in sorted(coverage_of(observations, family=family).items()):
            if coverage.attempted:
                rows.append((family, coverage))
    if not rows:
        return

    console.print("\n[bold]coverage[/bold]")
    for family, coverage in rows:
        note = " [red]LOW_COVERAGE[/red]" if coverage.low else ""
        console.print(
            f"  {family:16s} scored {coverage.scored}, "
            f"unscorable {coverage.unscorable}, skipped {coverage.skipped}{note}"
        )


def _assumptions(console: Console, result: object) -> None:
    """§8.6: discovery is a bootstrap the user can correct."""
    assumptions: Sequence[tuple[str, str, str]] = result.assumptions  # type: ignore[attr-defined]
    if not assumptions:
        return
    console.print("\n[yellow]assumptions you can correct[/yellow]")
    for field_name, confidence, detail in assumptions:
        console.print(f"  {field_name} ({confidence}) — {detail}")
