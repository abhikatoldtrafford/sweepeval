"""Sweep to frontier (spec §12, §14).

The one place that knows about both halves. It lives here rather than in
either because the layering contract keeps ``rank.domination`` unreachable
from ``execute`` — a domination decision made outside ``rank`` would not go
through Holm, and I2 would stop holding. So ``execute`` produces
measurements, ``rank`` decides the frontier, and this module carries plain
data between them.

Nothing composite is computed on the way through (I1). What crosses the
boundary is per-config cluster tables, intervals, coverage counts, and the
confirmed hard-fail census.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sweepeval.execute.sweep import SweepResult
from sweepeval.rank.constraints import DEFAULT_CONSTRAINTS, Constraint
from sweepeval.rank.coverage import coverage_from_counts
from sweepeval.rank.frontier import FrontierResult, rank_configs
from sweepeval.schema.objective import REGISTRY, Objective

__all__ = ["arun_and_rank", "rank_sweep", "select_objectives"]


def select_objectives(names: Sequence[str] | None = None) -> tuple[Objective, ...]:
    """The six defaults, or the user's narrowed set (§14.1).

    Narrowing offline is safe because every config runs the full profile —
    screening is cut (§12.6), so no objective was measured on a different
    probe set from the others.
    """
    if not names:
        return tuple(REGISTRY.defaults())

    from sweepeval.rank.prefer import resolve

    chosen: list[Objective] = []
    unknown: list[str] = []
    for name in names:
        try:
            chosen.append(REGISTRY.get(resolve(name)))
        except KeyError:
            unknown.append(name)
    if unknown:
        available = ", ".join(sorted(o.id for o in REGISTRY.all()))
        raise ValueError(
            f"unknown objective(s): {', '.join(unknown)}. Available: {available}"
        )
    return tuple(chosen)


def rank_sweep(
    result: SweepResult,
    *,
    objectives: Sequence[str] | None = None,
    alpha: float = 0.05,
    seed: int = 0,
    constraints: Sequence[Constraint] = DEFAULT_CONSTRAINTS,
) -> FrontierResult:
    """Compute the frontier over a completed (or partial) sweep."""
    config_ids = [row.config_id for row in result.configs]

    metrics = {row.config_id: row.metrics for row in result.configs}
    clusters = {row.config_id: row.clusters for row in result.configs}
    counts = {
        row.config_id: {"security_hard_fails": float(row.hard_fails.count)}
        for row in result.configs
    }

    # Strata are per metric and identical across configs by construction (the
    # probe set is frozen), so the first config's labels describe every one.
    strata: dict[str, dict[str, str]] = {}
    for row in result.configs:
        for metric, labels in row.strata.items():
            strata.setdefault(metric, {}).update(labels)

    # Counts, not observations: a stored run has aggregates and no log, and
    # offline re-ranking has to give the same answer as the live one.
    coverage = coverage_from_counts(
        {row.config_id: row.coverage for row in result.configs}
    )

    return rank_configs(
        config_ids,
        metrics,
        clusters,
        select_objectives(objectives),
        alpha=alpha,
        seed=seed,
        constraints=constraints,
        counts=counts,
        coverage=coverage,
        strata=strata,
    )


async def arun_and_rank(
    url: str, **kwargs: Any
) -> tuple[SweepResult, FrontierResult | None]:
    """The zero-config path: sweep, then rank what completed (§1).

    A sweep that sent nothing has nothing to rank, and returning an empty
    frontier for it would present "we did not run" as "nothing separated".
    """
    from sweepeval.execute.sweep import asweep_target

    objectives = kwargs.pop("objectives", None)
    alpha = kwargs.pop("alpha", 0.05)
    seed = kwargs.get("seed", 0)

    result = await asweep_target(url, **kwargs)
    if not result.configs:
        return result, None
    return result, rank_sweep(result, objectives=objectives, alpha=alpha, seed=seed)
