"""The frontier (spec §14). I1 and I2 load-bearing.

Puts §14's pieces in the order the spec requires and no other:

1. **Constraints first.** A config with a confirmed hard fail is disqualified
   before any comparison, because "slightly behind on security" is the wrong
   description of a leaked secret.
2. **Coverage parity.** A pair whose scored counts diverge is not compared;
   the paired test pairs on the probe, and a probe only one side scored is not
   a matched block.
3. **Paired tests, then Holm.** Domination is asserted only when Holm rejects
   at the family-wise error rate (I2). That decision is made in one module
   — ``rank.domination`` — and the layering contract enforces it.
4. **Complete-linkage tied clusters**, with spread.
5. **Correlation**, so a frontier that contains everything can be read for the
   reason it does.

No composite score is computed anywhere here, and none is stored (I1). The
only thing resembling an ordering is ``--prefer``, which applies the *user's*
declared preference offline and prints it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from sweepeval.rank.cluster import TieCluster, cluster_ties, spread_of
from sweepeval.rank.constraints import (
    DEFAULT_CONSTRAINTS,
    Constraint,
    ConstraintReport,
    apply_constraints,
)
from sweepeval.rank.coverage import CoverageMatrix
from sweepeval.rank.domination import DominationResult, compare_all, frontier_of
from sweepeval.schema.metric import Flag, MetricValue
from sweepeval.schema.objective import Objective
from sweepeval.stats.correlation import CorrelationMatrix, correlation_matrix
from sweepeval.stats.paired import PairedResult, paired_difference, resamples_for
from sweepeval.stats.resample import N_RESAMPLES
from sweepeval.stats.retention import auc_statistic

__all__ = [
    "FrontierResult",
    "make_paired",
    "rank_configs",
]

_QUANTILE_OBJECTIVES = {"latency_p95_ms": 0.95}

_CURVE_OBJECTIVES = frozenset({"context_retention_auc"})
"""Objectives whose statistic is a curve over strata, not a mean over
clusters. Their comparison has to recompute the curve on each replicate."""

COMPANION_METRICS: tuple[str, ...] = ("config_repeatability", "error_rate")
"""Reported beside the frontier without being on it (§14.2)."""

PairedFn = Callable[[str, str, Objective], PairedResult]


@dataclass
class FrontierResult:
    """The frontier and everything needed to read it."""

    objectives: tuple[Objective, ...] = ()
    frontier: tuple[str, ...] = ()
    dominated: dict[str, tuple[str, ...]] = field(default_factory=dict)
    clusters: tuple[TieCluster, ...] = ()
    domination: DominationResult | None = None
    constraints: ConstraintReport | None = None
    coverage: CoverageMatrix | None = None
    correlation: CorrelationMatrix = field(default_factory=CorrelationMatrix)
    blocked_pairs: dict[tuple[str, str], tuple[str, ...]] = field(default_factory=dict)
    excluded_metrics: dict[str, tuple[str, ...]] = field(default_factory=dict)
    points: dict[str, dict[str, float]] = field(default_factory=dict)
    companion_points: dict[str, dict[str, float]] = field(default_factory=dict)
    """Reported-but-not-selected metrics the report needs beside the frontier.

    ``config_repeatability`` above all: §14.2 requires both determinism
    numbers side by side whether or not the user promoted the second one, and
    reading it out of ``points`` would print an em dash for every row when it
    is not a selected objective.
    """

    eligible: tuple[str, ...] = ()
    alpha: float = 0.05

    @property
    def single_cluster(self) -> bool:
        """*Everything* tied — the frontier is the whole sweep and it is one
        cluster.

        Requiring the frontier to cover every eligible config matters: with
        two of four dominated, one cluster of two is a real separation, and
        saying "every configuration is statistically tied" there would report
        the opposite of what happened.
        """
        return (
            len(self.clusters) == 1
            and self.clusters[0].size > 1
            and len(self.frontier) == len(self.eligible)
        )

    def violators(self) -> tuple[str, ...]:
        if self.constraints is None:
            return ()
        return tuple(sorted(self.constraints.violators()))


def make_paired(
    clusters: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    seed: int = 0,
    alpha: float = 0.05,
    strata: Mapping[str, Mapping[str, str]] | None = None,
    n_resamples: int | None = None,
) -> PairedFn:
    """Build the paired-comparison function over per-config cluster tables.

    Only clusters **both** configs scored are compared. That intersection is
    the matched block I4 promises; taking the union and treating a missing
    cluster as zero would score a config for a probe it never answered.
    """
    strata = strata or {}

    def paired(a: str, b: str, objective: Objective) -> PairedResult:
        metric = objective.metric
        table_a = clusters.get(a, {}).get(metric, {})
        table_b = clusters.get(b, {}).get(metric, {})
        shared = sorted(set(table_a) & set(table_b))

        quantile = _QUANTILE_OBJECTIVES.get(objective.id)
        metric_strata = strata.get(metric) or {}

        if objective.id in _CURVE_OBJECTIVES and metric_strata:
            # The AUC is a depth-weighted trapezoid, not a mean. Comparing it
            # as a mean tests a different quantity from the one reported.
            statistic_a = auc_statistic(table_a, metric_strata)
            statistic_b = auc_statistic(table_b, metric_strata)
            quantile = None
            curve = True
        else:
            statistic_a = _statistic(table_a, quantile)
            statistic_b = _statistic(table_b, quantile)
            curve = False

        margin = _margin(objective, table_a, statistic_a)
        bounded = objective.min_effect_kind == "absolute"

        return paired_difference(
            shared,
            statistic_a,
            statistic_b,
            direction=objective.direction,
            margin=margin,
            alpha=alpha,
            seed=seed,
            strata=strata.get(metric),
            bounded=bounded,
            # The vectorised path: the same draw, computed as one matrix op so
            # the family-appropriate resample count is affordable.
            # The vectorised path assumes the statistic is separable per
            # cluster. A curve statistic is not, so it falls back to the loop.
            values_a=None if curve else table_a,
            values_b=None if curve else table_b,
            quantile=quantile,
            n_resamples=n_resamples or N_RESAMPLES,
        )

    return paired


def _statistic(
    table: Mapping[str, float], quantile: float | None
) -> Callable[[Sequence[str]], float]:
    def statistic(cluster_ids: Sequence[str]) -> float:
        values = [table[c] for c in cluster_ids if c in table]
        if not values:
            return 0.0
        if quantile is None:
            return sum(values) / len(values)
        ordered = sorted(values)
        # Nearest-rank, computed here rather than with numpy: the layering
        # contract keeps numpy out of rank, and a quantile of a short list has
        # no interesting numerics.
        index = min(len(ordered) - 1, round(quantile * (len(ordered) - 1)))
        return ordered[index]

    return statistic


def _margin(
    objective: Objective,
    table_a: Mapping[str, float],
    statistic_a: Callable[[Sequence[str]], float],
) -> float:
    """Absolute margins pass through; relative ones scale off the baseline.

    A 25% relative margin on latency has to be 25% *of something*, and the
    something is the reference config's own level **on the statistic actually
    being tested**. Scaling off the mean while comparing p95s made the margin
    roughly half what it should be, which biases toward asserting domination
    -- the unsafe direction for I2.
    """
    if objective.min_effect_kind != "relative":
        return objective.min_effect
    if not table_a:
        return objective.min_effect
    baseline = statistic_a(sorted(table_a))
    return abs(baseline) * objective.min_effect


def rank_configs(
    config_ids: Sequence[str],
    metrics: Mapping[str, Mapping[str, MetricValue]],
    clusters: Mapping[str, Mapping[str, Mapping[str, float]]],
    objectives: Sequence[Objective],
    *,
    alpha: float = 0.05,
    seed: int = 0,
    constraints: Sequence[Constraint] = DEFAULT_CONSTRAINTS,
    counts: Mapping[str, Mapping[str, float]] | None = None,
    coverage: CoverageMatrix | None = None,
    strata: Mapping[str, Mapping[str, str]] | None = None,
) -> FrontierResult:
    """Constraints, coverage, paired tests, Holm, clusters, correlation."""
    constraint_report = apply_constraints(
        config_ids, metrics, constraints, counts=counts
    )
    eligible = list(constraint_report.eligible)

    objectives = tuple(objectives)
    excluded = _excluded_metrics(eligible, metrics, objectives, coverage)
    blocked = _blocked_pairs(eligible, objectives, coverage)

    # Holm's tightest threshold is alpha / (ordered pairs), and the
    # superiority half is Bonferroni-multiplied by the objective count. The
    # bootstrap has to resolve finer than that or the family cannot reject at
    # all -- see resamples_for.
    pairs = max(1, len(eligible) * (len(eligible) - 1))
    n_resamples = resamples_for(pairs * max(1, len(objectives)), alpha)

    paired = make_paired(
        clusters, seed=seed, alpha=alpha, strata=strata, n_resamples=n_resamples
    )
    domination = compare_all(eligible, objectives, paired, alpha=alpha)

    # Coverage-blocked pairs are left in the Holm family rather than removed
    # from it. Removing them would shrink the family and loosen every other
    # pair's threshold, which can only ever create a domination that the full
    # family would not have licensed. I2 is one-directional: it forbids
    # asserting domination on weak evidence, never the reverse.
    for (a, b), families in blocked.items():
        verdict = domination.verdicts.get((a, b))
        if verdict is not None and verdict.dominates:
            verdict.dominates = False
            verdict.reason = (
                "not dominating: coverage diverges by more than 10% on "
                f"{', '.join(families)}, so these two were not scored on the "
                "same probes and the pairing does not hold (§14.5)"
            )

    frontier, dominated = frontier_of(eligible, domination)
    points = _points(eligible, metrics, objectives)

    clusters_out = tuple(
        spread_of(cluster, points, [o.id for o in objectives])
        for cluster in cluster_ties(frontier, domination.tied)
    )

    companion = {
        config_id: {
            metric: value.point
            for metric in COMPANION_METRICS
            if (value := metrics.get(config_id, {}).get(metric)) is not None
            and Flag.NO_VALID_INTERVAL not in value.flags
        }
        for config_id in eligible
    }

    return FrontierResult(
        objectives=objectives,
        frontier=frontier,
        dominated=dict(dominated),
        clusters=clusters_out,
        domination=domination,
        constraints=constraint_report,
        coverage=coverage,
        correlation=correlation_matrix(points, [o.id for o in objectives]),
        blocked_pairs=blocked,
        excluded_metrics=excluded,
        points=points,
        companion_points=companion,
        eligible=tuple(eligible),
        alpha=alpha,
    )


def _points(
    config_ids: Sequence[str],
    metrics: Mapping[str, Mapping[str, MetricValue]],
    objectives: Sequence[Objective],
) -> dict[str, dict[str, float]]:
    """Point estimates per config, for spread and correlation only.

    Never for a comparison: every comparison goes through the paired
    bootstrap. A metric with no valid interval contributes no point, because
    its ``point`` is a placeholder and reading it as a measurement is the bug
    the em dash in the terminal report exists to prevent.
    """
    out: dict[str, dict[str, float]] = {}
    for config_id in config_ids:
        row = metrics.get(config_id, {})
        values: dict[str, float] = {}
        for objective in objectives:
            value = row.get(objective.metric)
            if value is None or Flag.NO_VALID_INTERVAL in value.flags:
                continue
            values[objective.id] = value.point
        out[config_id] = values
    return out


def _blocked_pairs(
    config_ids: Sequence[str],
    objectives: Sequence[Objective],
    coverage: CoverageMatrix | None,
) -> dict[tuple[str, str], tuple[str, ...]]:
    if coverage is None:
        return {}
    families = {o.family for o in objectives}
    blocked: dict[tuple[str, str], tuple[str, ...]] = {}
    for a in config_ids:
        for b in config_ids:
            if a == b:
                continue
            diverged = tuple(
                f for f in coverage.blocked_families(a, b) if f in families
            )
            if diverged:
                blocked[(a, b)] = diverged
    return blocked


def _excluded_metrics(
    config_ids: Sequence[str],
    metrics: Mapping[str, Mapping[str, MetricValue]],
    objectives: Sequence[Objective],
    coverage: CoverageMatrix | None,
) -> dict[str, tuple[str, ...]]:
    """Objectives dropped for a config, with the reason (§14.5, §11.8)."""
    out: dict[str, tuple[str, ...]] = {}
    for config_id in config_ids:
        row = metrics.get(config_id, {})
        dropped: list[str] = []
        for objective in objectives:
            value = row.get(objective.metric)
            if value is None or Flag.LOW_COVERAGE in value.flags:
                dropped.append(objective.id)
                continue
            if coverage is not None and any(
                coverage.excludes(config_id, other, objective.family)
                for other in config_ids
                if other != config_id
            ):
                dropped.append(objective.id)
        if dropped:
            out[config_id] = tuple(dropped)
    return out
