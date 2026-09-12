"""Frontier reporting (spec §14.2, §14.3, §14.6, §15). I1 load-bearing.

What a frontier report must always carry, because the number alone is not
interpretable:

* the clusters, with their **spread** — members are statistically
  indistinguishable, and where the observed range is nonetheless wide the
  report says the fix is more runs, not a closer reading;
* what each cluster **wins and gives up**, as ranges across members;
* every **violator** and the constraint it broke, so the frontier reconciles
  with the measurements table;
* the **objective correlation matrix**, so a frontier that contains everything
  can be read for the reason it does (§14.3);
* **both determinism numbers side by side** — the temp-0 objective with its
  (model, system_prompt) scope stated, and ``config_repeatability`` as the
  production-truth figure (§14.2);
* at ``quick``, that intervals are wide, few pairs will separate, and results
  are **not gate-eligible** — which is different from, and must never be
  written as, "no valid intervals".

Nothing composite is printed. There is no rank column, no score, and no
ordering except the one ``--prefer`` produces from the user's own stated
preference.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

__all__ = ["render_frontier", "render_preference"]

_DETERMINISM = ("target_determinism_at_temp0", "config_repeatability")


def render_frontier(
    result: Any, labels: dict[str, str] | None = None, console: Console | None = None
) -> None:
    """Render a computed frontier. ``result`` is a ``rank.FrontierResult``."""
    console = console or Console()
    labels = labels or {}

    def name(config_id: str) -> str:
        return f"{config_id}  {labels[config_id]}" if config_id in labels else config_id

    _violators(console, result)
    _clusters(console, result, name)
    _dominated(console, result, name)
    _determinism(console, result)
    _correlation(console, result)
    _blocked(console, result)
    _weights(console, result)


def _weights(console: Console, result: Any) -> None:
    """I1: within-family aggregation weights are published, never implied.

    ``security_pass_rate`` weights all eight attack classes equally despite
    their differing declared severity; severity drives hard-fail
    classification, not weighting. Stating that is the difference between a
    weighting the reader can disagree with and one they cannot see.
    """
    if not result.objectives:
        return
    console.print("\n[bold]how each objective is aggregated[/bold] (I1)")
    for objective in result.objectives:
        console.print(f"  {objective.id:32s} {objective.weighting}")
        if objective.note:
            console.print(f"  {'':32s} [dim]{objective.note}[/dim]")


def _violators(console: Console, result: Any) -> None:
    """§14.1: excluded before the frontier, and never silently."""
    report = result.constraints
    if report is None or not report.violations:
        return
    console.print("\n[red]excluded by a hard constraint[/red] (before any comparison)")
    for violation in report.violations:
        console.print(f"  {violation.describe()}")


def _clusters(console: Console, result: Any, name) -> None:
    if not result.clusters:
        console.print("\n[red]no configuration reached the frontier[/red]")
        return

    console.print(
        f"\n[bold]frontier[/bold] — {len(result.frontier)} non-dominated config(s) "
        f"in {len(result.clusters)} tied cluster(s), at alpha {result.alpha:g} "
        "family-wise"
    )

    for index, cluster in enumerate(result.clusters, start=1):
        console.print(f"\n  [bold]cluster {index}[/bold] ({cluster.size} config(s))")
        for member in cluster.members:
            console.print(f"    {name(member)}")
        if not cluster.spread:
            continue
        console.print("    spread across members:")
        for objective in sorted(cluster.spread):
            low, high = cluster.spread[objective]
            wide = " [yellow]wide[/yellow]" if objective in cluster.wide else ""
            console.print(f"      {objective:32s} {low:.4g} .. {high:.4g}{wide}")
        if cluster.wide:
            console.print(
                "    [yellow]the range above is wide although no pair separates "
                "statistically — raise --runs, or use --profile standard[/yellow]"
            )

    if result.single_cluster:
        console.print(
            "\n  [yellow]every configuration is statistically tied.[/yellow] That is "
            "an answer, not a\n  failure: at this profile the intervals are wide "
            "enough that the sweep\n  cannot separate them. Use --prefer to apply "
            "your own priority, or\n  --profile standard for narrower intervals."
        )


def _dominated(console: Console, result: Any, name) -> None:
    if not result.dominated:
        return
    console.print("\n[bold]dominated[/bold]")
    for config_id, dominators in sorted(result.dominated.items()):
        console.print(f"  {name(config_id)}")
        console.print(f"    dominated by {', '.join(dominators)}")
        # The verdict is keyed (dominated, dominator), so its wins and
        # concedes belong to the DOMINATOR. Printing them under the dominated
        # config's name without saying whose they are reads as the dominated
        # config winning the objective it lost on.
        first = dominators[0]
        verdict = _verdict(result, config_id, first)
        if verdict is not None:
            if verdict.wins:
                console.print(
                    f"    {first} is better on: {', '.join(verdict.wins)}"
                )
            if verdict.concedes:
                console.print(
                    f"    {first} gives up:     {', '.join(verdict.concedes)}"
                )
            else:
                console.print(f"    {first} gives up:     nothing")
            if verdict.adjusted_threshold is not None:
                console.print(
                    f"    p={verdict.p_pair:.4g} against a Holm threshold of "
                    f"{verdict.adjusted_threshold:.4g}"
                )


def _verdict(result: Any, a: str, b: str) -> Any:
    if result.domination is None:
        return None
    return result.domination.verdicts.get((a, b))


def _determinism(console: Console, result: Any) -> None:
    """§14.2: both numbers, side by side, with the confound stated."""
    ids = {o.id for o in result.objectives}
    if "target_determinism_at_temp0" not in ids:
        return
    rows = [
        (c, result.points.get(c, {})) for c in sorted(result.points)
    ]
    if not rows:
        return

    table = Table(title="determinism, both ways (§14.2)", header_style="bold")
    table.add_column("config")
    table.add_column("at temp=0", justify="right")
    table.add_column("at its own settings", justify="right")
    shown = False
    for config_id, points in rows:
        at_zero = points.get("target_determinism_at_temp0")
        own = result.companion_points.get(config_id, {}).get("config_repeatability")
        if at_zero is None and own is None:
            continue
        shown = True
        table.add_row(
            config_id,
            "—" if at_zero is None else f"{at_zero:.4g}",
            "—" if own is None else f"{own:.4g}",
        )
    if not shown:
        return

    console.print()
    console.print(table)
    console.print(
        "  at temp=0 is a property of the (model, system_prompt) pair, shared "
        "across that\n  pair's temperature siblings — it does not describe the "
        "row's own sampling.\n  at its own settings is the production-truth "
        "figure; promote it with --objective\n  config_repeatability if that is "
        "the one you want on the frontier."
    )


def _correlation(console: Console, result: Any) -> None:
    """§14.3: six objectives probably span fewer than six dimensions."""
    matrix = result.correlation
    if not matrix.computed:
        if matrix.reason:
            console.print(f"\n[dim]objective correlation: {matrix.reason}[/dim]")
        return

    console.print("\n[bold]objective correlation[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("")
    for objective in matrix.objectives:
        table.add_column(_abbrev(objective), justify="right")
    for a in matrix.objectives:
        table.add_row(
            _abbrev(a),
            *(f"{matrix.get(a, b) or 0.0:+.2f}" for b in matrix.objectives),
        )
    console.print(table)

    for a, b, value in matrix.high_pairs():
        console.print(
            f"  [yellow]{a} and {b} correlate at {value:+.2f}[/yellow] — they are "
            f"close to measuring one thing.\n  Drop one with --objectives to stop "
            "it adding a dimension nothing can be dominated in."
        )


def _abbrev(objective: str) -> str:
    return {
        "security_pass_rate": "sec",
        "guardrail_pass_rate": "guard",
        "target_determinism_at_temp0": "det@0",
        "config_repeatability": "repeat",
        "context_retention_auc": "retain",
        "latency_p95_ms": "lat",
        "cost_per_probe": "cost",
    }.get(objective, objective[:8])


def _blocked(console: Console, result: Any) -> None:
    """§14.5: a pair that was not compared says so."""
    if not result.blocked_pairs:
        return
    shown = sorted({tuple(sorted(pair)) for pair in result.blocked_pairs})
    console.print(
        f"\n[yellow]{len(shown)} pair(s) were not compared[/yellow] — coverage "
        "diverges by more than 10%,\nso the two were not scored on the same "
        "probes and the pairing does not hold:"
    )
    for a, b in shown[:6]:
        families = result.blocked_pairs.get((a, b)) or result.blocked_pairs.get((b, a))
        console.print(f"  {a} vs {b}: {', '.join(families or ())}")
    if len(shown) > 6:
        console.print(f"  ... and {len(shown) - 6} more")


def render_preference(
    outcome: Any, labels: dict[str, str] | None = None, console: Console | None = None
) -> None:
    """§14.4: name one config, show what it concedes, print the preference."""
    console = console or Console()
    labels = labels or {}

    console.print(
        Panel(
            f"[bold]{outcome.preference.describe()}[/bold]\n"
            f"{outcome.reason}",
            title="your preference",
            expand=False,
        )
    )

    if outcome.chosen is None:
        return

    label = labels.get(outcome.chosen, "")
    console.print(f"\n  [bold green]{outcome.chosen}[/bold green]  {label}")

    if outcome.concedes:
        console.print("\n  it concedes:")
        for objective, better in sorted(outcome.concedes.items()):
            console.print(f"    {objective:32s} beaten by {', '.join(better)}")
    else:
        console.print(
            "\n  it concedes nothing on the frontier's objectives at this alpha."
        )

    if outcome.excluded:
        console.print("\n  excluded by your constraint:")
        for config_id, bound in sorted(outcome.excluded.items()):
            console.print(f"    {config_id:10s} failed {bound}")

    console.print(
        "\n  [dim]This is your preference applied to stored results, not a score "
        "the tool\n  computed. Change it and re-report; no re-run is needed.[/dim]"
    )
