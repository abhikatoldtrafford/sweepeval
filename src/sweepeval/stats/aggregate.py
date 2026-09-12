"""Aggregation from observations to metrics (spec §13, §6.4).

Turns the append-only observation log into per-config ``MetricValue``s. Three
things this layer is responsible for getting right:

* **Clustering.** Runs within a probe are averaged *before* the probe becomes a
  cluster. Treating probexrun as independent trials would inflate the effective
  n by the run count and understate every interval — at temp=0 the three runs
  are often byte-identical, so the true information is one probe, not three.
* **Exclusions.** ``UNSCORABLE`` trials leave the numerator and the
  denominator (§11.8). A refused determinism trial is not a failed one.
* **Coverage.** What was excluded is counted and reported, and a family losing
  more than 30% of its trials is flagged ``LOW_COVERAGE`` rather than quietly
  reported as if nothing were missing.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from sweepeval.schema.metric import Estimand, Flag, MetricValue
from sweepeval.schema.observation import Observation, Verdict
from sweepeval.stats.resample import CLUSTER_FLOOR, cluster_bootstrap

__all__ = [
    "LOW_COVERAGE_THRESHOLD",
    "ClusterTable",
    "Coverage",
    "aggregate_metric",
    "build_cluster_table",
    "coverage_of",
]

LOW_COVERAGE_THRESHOLD = 0.30
"""§11.8: above this fraction unscorable, the metric is flagged."""


@dataclass(frozen=True)
class Coverage:
    scored: int
    unscorable: int
    skipped: int
    reasons: tuple[str, ...] = ()

    @property
    def attempted(self) -> int:
        return self.scored + self.unscorable + self.skipped

    @property
    def unscorable_fraction(self) -> float:
        return self.unscorable / self.attempted if self.attempted else 0.0

    @property
    def low(self) -> bool:
        return self.unscorable_fraction > LOW_COVERAGE_THRESHOLD


@dataclass
class ClusterTable:
    """Per-cluster values for one config and metric.

    The cluster is the probe (or, for retention, the conversation). Values are
    already averaged over runs, which is what makes the probe the unit of
    replication rather than probexrun.
    """

    values: dict[str, float] = field(default_factory=dict)
    strata: dict[str, str] = field(default_factory=dict)
    coverage: Coverage = field(default_factory=lambda: Coverage(0, 0, 0))

    @property
    def cluster_ids(self) -> list[str]:
        return sorted(self.values)

    def statistic(self) -> Callable[[Sequence[str]], float]:
        """A mean-over-drawn-clusters statistic for the bootstrap."""

        def inner(drawn: Sequence[str]) -> float:
            return sum(self.values[c] for c in drawn) / len(drawn)

        return inner


def build_cluster_table(
    observations: Iterable[Observation],
    *,
    metric: str,
    config_id: str,
    stratify_by_depth: bool = False,
) -> ClusterTable:
    """Collapse observations into one value per cluster.

    ``UNSCORABLE`` and ``SKIPPED`` trials are excluded from the value and
    counted in coverage. A cluster whose every trial was excluded does not
    appear at all — it is absent, not zero, and a zero would read as a failure.
    """
    per_cluster: dict[str, list[float]] = defaultdict(list)
    strata: dict[str, str] = {}
    scored = unscorable = skipped = 0
    reasons: set[str] = set()

    for observation in observations:
        if observation.metric != metric or observation.config_id != config_id:
            continue

        if observation.verdict is Verdict.SKIPPED:
            skipped += 1
            if observation.reason:
                reasons.add(observation.reason)
            continue
        if observation.verdict is Verdict.UNSCORABLE:
            unscorable += 1
            if observation.reason:
                reasons.add(observation.reason)
            continue

        value = observation.value
        if value is None:
            value = 1.0 if observation.verdict is Verdict.PASS else 0.0

        per_cluster[observation.unit_id].append(value)
        scored += 1
        if stratify_by_depth and observation.depth is not None:
            strata[observation.unit_id] = f"d{observation.depth}"

    return ClusterTable(
        # Runs averaged within the probe: the probe is the cluster, not
        # probe x run.
        values={cid: sum(v) / len(v) for cid, v in per_cluster.items()},
        strata=strata if stratify_by_depth else {},
        coverage=Coverage(scored, unscorable, skipped, tuple(sorted(reasons))),
    )


def aggregate_metric(
    table: ClusterTable,
    *,
    estimand: Estimand = Estimand.generalization,
    alpha: float = 0.05,
    seed: int = 0,
    bounded: bool = True,
    indicative: bool = False,
) -> MetricValue:
    """One config's value for one metric, with its interval.

    ``indicative`` is set for the quick profile and per-cell breakdowns: the
    interval is valid but wide, which is a different state from having no valid
    interval at all (§6.4).
    """
    flags: list[Flag] = []
    if indicative:
        flags.append(Flag.INDICATIVE)
    if table.coverage.low:
        flags.append(Flag.LOW_COVERAGE)

    ids = table.cluster_ids
    if not ids:
        from sweepeval.stats.resample import no_valid_interval

        return no_valid_interval(
            0.0, 0, alpha, estimand,
            extra_flags=(*flags, Flag.LOW_N),
        )

    return cluster_bootstrap(
        ids,
        table.statistic(),
        estimand=estimand,
        alpha=alpha,
        seed=seed,
        strata=table.strata or None,
        bounded=bounded,
        extra_flags=tuple(flags),
    )


def coverage_of(
    observations: Iterable[Observation], *, family: str
) -> Mapping[str, Coverage]:
    """Per-config coverage for one family, for the report's coverage block."""
    buckets: dict[str, list[Observation]] = defaultdict(list)
    for observation in observations:
        if observation.family == family:
            buckets[observation.config_id].append(observation)

    out: dict[str, Coverage] = {}
    for config_id, rows in buckets.items():
        scored = sum(1 for r in rows if r.verdict in (Verdict.PASS, Verdict.FAIL))
        unscorable = sum(1 for r in rows if r.verdict is Verdict.UNSCORABLE)
        skipped = sum(1 for r in rows if r.verdict is Verdict.SKIPPED)
        reasons = sorted({r.reason for r in rows if r.reason})
        out[config_id] = Coverage(scored, unscorable, skipped, tuple(reasons))
    return out


def below_floor(table: ClusterTable) -> bool:
    """Whether this table cannot support a bootstrap interval (§13.4)."""
    return len(table.values) < CLUSTER_FLOOR


__all__ += ["below_floor"]
