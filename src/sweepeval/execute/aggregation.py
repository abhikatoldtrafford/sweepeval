"""Per-configuration aggregation (spec §13.3, §11.5, I5).

One implementation, used by both the single-config evaluation and every row of
a sweep. Two implementations would drift, and the way they would drift is that
a sweep's numbers stop matching the evaluation's for the same target — which
is unfalsifiable from the outside, because both look plausible.

I5 lives here: a metric whose every trial was unscorable is still emitted,
flagged ``NO_VALID_INTERVAL``. A metric that vanishes from a report is
indistinguishable from one that passed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from sweepeval.execute.cost import Pricing
from sweepeval.schema.metric import Estimand, Flag, MetricValue
from sweepeval.schema.observation import Observation
from sweepeval.stats.aggregate import (
    ClusterTable,
    aggregate_metric,
    build_cluster_table,
)
from sweepeval.stats.resample import order_statistic_interval
from sweepeval.stats.retention import auc_statistic

__all__ = ["RATE_METRICS", "ConfigAggregate", "aggregate_config"]

RATE_METRICS: tuple[str, ...] = (
    "security_pass_rate",
    "guardrail_pass_rate",
    "config_repeatability",
    "target_determinism_at_temp0",
    "semantic_stability",
    "invariance",
    "degradation_resilience",
    "load_resilience",
)

LATENCY_OBJECTIVE = "latency_p95_ms"
LATENCY_QUANTILE = 0.95

LATENCY_FALLBACK_OBJECTIVE = "latency_p90_ms"
LATENCY_FALLBACK_QUANTILE = 0.90
"""§13.3's documented fallback, finally emitted.

A distribution-free upper bound for a p95 needs 72 probes; a p90 needs 36. At
`standard` (69 probes) the p95 cannot carry an honest interval and the p90
can, which is precisely the case the spec wrote the fallback for."""
"""§14.1's latency objective: the 95th percentile of per-probe mean latency.

The resampling unit is the probe, so the cluster value is that probe's mean
call latency and the statistic is a quantile *across* probes. Stating the
composition matters -- it is not a quantile over raw calls, and §13.3's
fallback to p90 exists because the p95 of a short cluster list is noisy.
"""

_OPERATIONAL: tuple[tuple[str, bool], ...] = (
    ("latency_ms", False),
    ("tokens_out", False),
    ("error_rate", True),
)


@dataclass
class ConfigAggregate:
    """Everything the ranker needs about one configuration."""

    config_id: str
    metrics: dict[str, MetricValue] = field(default_factory=dict)
    clusters: dict[str, dict[str, float]] = field(default_factory=dict)
    """Per-metric cluster values, kept because the paired test needs the
    matched blocks, not the summary (§13.3)."""

    strata: dict[str, dict[str, str]] = field(default_factory=dict)
    """Per-metric stratum labels, carried through for the same reason.

    Retention resamples within depth; an unstratified resample empties a depth
    in a few percent of replicates and the trapezoid is undefined there
    (§13.3). Dropping the labels after aggregation would silently unstratify
    every paired comparison downstream.
    """

    retention_curve: dict[int, float] = field(default_factory=dict)
    retention_weights: dict[int, float] = field(default_factory=dict)
    retention_depth_at_floor: int | None = None


def aggregate_config(
    observations: Sequence[Observation],
    *,
    config_id: str,
    seed: int = 0,
    indicative: bool = False,
    pricing: Pricing | None = None,
) -> ConfigAggregate:
    """Aggregate one config's observations into intervals."""
    out = ConfigAggregate(config_id=config_id)

    for metric in RATE_METRICS:
        table = build_cluster_table(observations, metric=metric, config_id=config_id)
        out.clusters[metric] = dict(table.values)
        # Emitted even with no scored cluster (I5).
        out.metrics[metric] = aggregate_metric(
            table, estimand=Estimand.generalization, seed=seed, indicative=indicative
        )

    _context(out, observations, seed=seed, indicative=indicative)
    _operational(out, observations, seed=seed, indicative=indicative)
    _cost(out, observations, seed=seed, indicative=indicative, pricing=pricing)
    return out


def _latency_quantiles(
    out: ConfigAggregate, table: ClusterTable, *, indicative: bool
) -> None:
    """Tail latency, at whatever quantile this many probes can actually bound.

    A bootstrap percentile interval under-covers badly for a quantile: the
    p95's measured coverage was 0.713 at 24 clusters and 0.883 at 69, printed
    beside a nominal 0.95. `order_statistic_interval` is exact, and declines
    when no upper bound exists at the sample size -- which for a p95 is
    anything below 72 clusters.

    So the p95 is emitted whenever the corpus can carry it and declines an
    interval otherwise (I3, never a bare point), and §13.3's documented
    fallback to p90 -- which needs 36 -- is emitted alongside it whenever
    *that* is estimable. Under its own name: reporting a p90 as
    `latency_p95_ms` is the class of defect this whole change is about.
    """
    values = list(table.values.values())
    extra = (Flag.INDICATIVE,) if indicative else ()

    for objective, quantile in (
        (LATENCY_OBJECTIVE, LATENCY_QUANTILE),
        (LATENCY_FALLBACK_OBJECTIVE, LATENCY_FALLBACK_QUANTILE),
    ):
        out.clusters[objective] = dict(table.values)
        out.metrics[objective] = order_statistic_interval(
            values, quantile, extra_flags=extra
        )


def _cost(
    out: ConfigAggregate,
    observations: Sequence[Observation],
    *,
    seed: int,
    indicative: bool,
    pricing: Pricing | None,
) -> None:
    """Per-probe spend, when the user supplied prices (§12.4, D8).

    Without pricing the cost objective degrades to output tokens, which the
    registry says and `pricing_source` makes a hard comparability key. WITH
    pricing it still ranked `tokens_out`: the dollars were computed for the
    printed cost block and never reached the frontier, so the axis labelled
    "Cost per probe" preferred a reasoning model at $0.021 over a plain one at
    $0.006. Input and reasoning tokens were invisible to it entirely.
    """
    if pricing is None:
        return

    tables = {
        name: build_cluster_table(observations, metric=name, config_id=out.config_id)
        for name in ("tokens_in", "tokens_out", "tokens_reasoning")
    }
    if not tables["tokens_out"].coverage.attempted:
        return

    # Reasoning tokens bill at the output rate on every provider that reports
    # them separately, and they are the bulk of a reasoning model's spend --
    # gpt-5-nano burned 576 of them for a one-line refusal.
    shared = set(tables["tokens_out"].values)
    for name in ("tokens_in", "tokens_reasoning"):
        if tables[name].coverage.attempted:
            shared &= set(tables[name].values)

    costs = {
        cluster: pricing.cost(
            int(tables["tokens_in"].values.get(cluster, 0.0)),
            int(tables["tokens_out"].values.get(cluster, 0.0))
            + int(tables["tokens_reasoning"].values.get(cluster, 0.0)),
        )
        for cluster in sorted(shared)
    }
    if not costs:
        return

    out.clusters["cost_usd"] = costs
    out.metrics["cost_usd"] = aggregate_metric(
        replace(tables["tokens_out"], values=costs),
        seed=seed,
        indicative=indicative,
    )


def _context(
    out: ConfigAggregate,
    observations: Sequence[Observation],
    *,
    seed: int,
    indicative: bool,
) -> None:
    """Recall per conversation, plus the depth-weighted AUC (§11.5)."""
    from sweepeval.stats.retention import (
        depth_at_floor,
        retention_auc,
        retention_curve,
    )

    table = build_cluster_table(
        observations,
        metric="fact_recall",
        config_id=out.config_id,
        stratify_by_depth=True,
    )
    if not table.coverage.attempted:
        return

    out.clusters["fact_recall"] = dict(table.values)
    out.strata["fact_recall"] = dict(table.strata)
    # Stratified: an unstratified resample can empty a depth, and the
    # trapezoid is undefined there (§13.3).
    out.metrics["fact_recall"] = aggregate_metric(
        table, seed=seed, indicative=indicative
    )

    curve = retention_curve(observations)
    _auc, weights = retention_auc(curve)
    out.retention_curve = curve
    out.retention_weights = weights
    out.retention_depth_at_floor = depth_at_floor(curve)

    # The AUC resamples conversations and recomputes the weighted curve on
    # each replicate, so the interval belongs to the AUC rather than to mean
    # recall. Stratified by depth: an unstratified resample can empty a depth
    # and the trapezoid is undefined there (§13.3).
    out.clusters["context_retention_auc"] = dict(table.values)
    out.strata["context_retention_auc"] = dict(table.strata)
    out.metrics["context_retention_auc"] = aggregate_metric(
        table,
        seed=seed,
        indicative=indicative,
        statistic=auc_statistic(table.values, table.strata),
    )


def _operational(
    out: ConfigAggregate,
    observations: Sequence[Observation],
    *,
    seed: int,
    indicative: bool,
) -> None:
    for metric, bounded in _OPERATIONAL:
        table = build_cluster_table(
            observations, metric=metric, config_id=out.config_id
        )
        if not table.coverage.attempted:
            continue
        out.clusters[metric] = dict(table.values)
        out.metrics[metric] = aggregate_metric(
            table, seed=seed, bounded=bounded, indicative=indicative
        )

        # §14.1's latency objective is a quantile, not a mean. It was reported
        # as the mean of per-probe latencies under the name latency_p95_ms
        # while the paired test compared a p95 -- the point, the interval and
        # the tested statistic were three different quantities sharing a name.
        # Emitted here as its own metric so all three agree.
        if metric == "latency_ms":
            _latency_quantiles(out, table, indicative=indicative)
