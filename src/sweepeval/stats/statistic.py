"""Which statistic an objective is compared with (spec §13.3, §16).

One dispatch, shared. There were two, and they disagreed.

`rank/frontier.py` learned that `context_retention_auc` is a depth-weighted
trapezoid and `latency_p95_ms` is a quantile; its own comment says "comparing
it as a mean tests a different quantity from the one reported". `stats/diff.py`
-- the CI gate -- took the unweighted mean of every objective, and
`context_retention_auc` is in `DEFAULT_GATE_ON`. On the same data:

    true depth-weighted AUC   0.6458 -> 0.3542   a 29-point regression
    what the gate compared    0.5000 -> 0.5000   "ok", exit 0

A model that keeps shallow recall and forgets mid-conversation passed CI
silently, and the two numbers the gate printed were means under the AUC's
name. `latency_p95_ms` was the same story: 300 ms to 2000 ms at p95 reported
as an improvement.

Keeping the dispatch in `stats` rather than `rank` is what lets the gate reach
it: `rank` may import `stats`, never the reverse.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from sweepeval.schema.objective import Objective
from sweepeval.stats.retention import auc_statistic

__all__ = [
    "CURVE_OBJECTIVES",
    "QUANTILE_OBJECTIVES",
    "ClusterStatistic",
    "mean_statistic",
    "quantile_statistic",
    "statistic_for",
    "usable_strata",
]

QUANTILE_OBJECTIVES: dict[str, float] = {"latency_p95_ms": 0.95}
"""Objectives reported as a quantile of the cluster values, not their mean."""

CURVE_OBJECTIVES = frozenset({"context_retention_auc"})
"""Objectives whose statistic is a curve over strata rather than a summary of
clusters. Their comparison has to recompute the curve on each replicate."""


@dataclass(frozen=True)
class ClusterStatistic:
    """How one objective turns a set of clusters into a number."""

    fn: Callable[[Sequence[str]], float]
    quantile: float | None = None
    curve: bool = False
    """True when the statistic reads strata, which the resampler must then
    stratify on -- an unstratified replicate can empty a depth, and the
    trapezoid is undefined there."""


def mean_statistic(table: Mapping[str, float]) -> Callable[[Sequence[str]], float]:
    def statistic(cluster_ids: Sequence[str]) -> float:
        values = [table[c] for c in cluster_ids if c in table]
        return sum(values) / len(values) if values else 0.0

    return statistic


def quantile_statistic(
    table: Mapping[str, float], quantile: float
) -> Callable[[Sequence[str]], float]:
    def statistic(cluster_ids: Sequence[str]) -> float:
        values = sorted(table[c] for c in cluster_ids if c in table)
        if not values:
            return 0.0
        # Nearest-rank, computed without numpy: the layering contract keeps
        # numpy out of `rank`, and a quantile of a short list has no
        # interesting numerics.
        index = min(len(values) - 1, round(quantile * (len(values) - 1)))
        return values[index]

    return statistic


def usable_strata(
    strata: Mapping[str, str] | None, table: Mapping[str, float]
) -> bool:
    """Do these labels actually name a depth the curve can be built from?

    `auc_statistic` skips any label not shaped ``d<int>`` and returns 0.0 when
    that leaves nothing -- so a strata map in the wrong format made the AUC
    0.0 on *both* sides of a comparison, which reads as "no change" and passes
    every gate. Harmless while strata existed only in memory; `Baseline.strata`
    puts them in a file a user can commit, diff and hand-edit.

    One parseable label is enough: a partly-labelled map is a coverage
    question, not a format question, and the curve handles it.
    """
    if not strata:
        return False
    return any(
        label.startswith("d") and label[1:].isdigit()
        for cluster, label in strata.items()
        if cluster in table
    )


def statistic_for(
    objective: Objective,
    table: Mapping[str, float],
    strata: Mapping[str, str] | None = None,
) -> ClusterStatistic:
    """The statistic this objective is *reported* with.

    ``strata`` is the metric's own stratum labels. A curve objective without
    them falls back to the mean -- a baseline written before `Baseline.strata`
    existed carries none, and refusing to gate at all would be worse than
    gating on the mean while saying so. Callers that can tell the user check
    :attr:`ClusterStatistic.curve` to find out which happened.
    """
    if objective.id in CURVE_OBJECTIVES and usable_strata(strata, table):
        return ClusterStatistic(fn=auc_statistic(table, strata or {}), curve=True)

    quantile = QUANTILE_OBJECTIVES.get(objective.id)
    if quantile is not None:
        return ClusterStatistic(fn=quantile_statistic(table, quantile), quantile=quantile)

    return ClusterStatistic(fn=mean_statistic(table))
