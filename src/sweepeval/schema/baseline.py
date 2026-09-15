"""The committable baseline (spec §16).

Holds comparability keys plus every metric's **per-cluster** values, not just a
point and an interval. The paired test needs them: the reason the gate is not
flappy is that the same clusters appear on both sides of the comparison, and a
summary statistic throws that away.

This file is designed to be committed to a user's repository, so nothing in it
may carry a credential — the redactor runs over everything written here
(§6.6), and ``sweepeval init`` tells the user what is safe to commit.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from sweepeval.schema.comparability import Comparability
from sweepeval.schema.metric import MetricValue
from sweepeval.schema.versions import SCHEMA_VERSION, TOOL_VERSION

__all__ = ["Baseline"]


class Baseline(BaseModel):
    """A snapshot of one run, for the gate to compare against."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    tool_version: str = TOOL_VERSION
    run_id: str
    created_at: str
    config_id: str
    comparability: Comparability

    metrics: dict[str, MetricValue]
    """Point and interval per metric, for display."""

    clusters: dict[str, dict[str, float]]
    """Per-cluster values per metric, for the paired test."""

    strata: dict[str, dict[str, str]] = Field(default_factory=dict)
    """Per-cluster stratum labels per metric -- conversation depth, in
    practice.

    Without these the gate cannot compute `context_retention_auc`, which is a
    depth-weighted trapezoid and is gated by default. It fell back to an
    unweighted mean, so a 29-point AUC regression read as 0.5000 -> 0.5000 and
    exited 0. Defaulted rather than required: a baseline committed before this
    field existed must still load, and the gate says when it degraded.
    """

    hard_fails: tuple[str, ...] = ()

    model: str | None = None
    """The model id the run's requests named, when they named one (§16).

    Deliberately not a comparability key. A sweep varies the model across
    configs within one run and the manifest carries one comparability block
    for the whole run, so a per-run key would have to lie for every sweep.
    This is the config's own parameter, recorded here because a baseline is a
    committed, shared file that said nothing about what produced it -- and
    because the model is chosen by a discovery heuristic when it is not
    pinned, so it can move between the baseline run and the gate run on its
    own.

    Defaulted rather than required: baselines committed before this field
    existed must still load, and the gate says when it could not check.
    """

    @field_serializer("clusters")
    def _sorted_clusters(
        self, value: dict[str, dict[str, float]]
    ) -> dict[str, dict[str, float]]:
        """Sorted, so a committed baseline diffs cleanly and hashes stably."""
        return {
            metric: dict(sorted(values.items()))
            for metric, values in sorted(value.items())
        }

    @field_serializer("metrics")
    def _sorted_metrics(self, value: dict[str, MetricValue]) -> dict[str, Any]:
        return {k: value[k].model_dump(mode="json") for k in sorted(value)}

    def shared_clusters(self, other: Baseline, metric: str) -> list[str]:
        """Clusters present in both. The paired test runs over exactly these."""
        mine = self.clusters.get(metric, {})
        theirs = other.clusters.get(metric, {})
        return sorted(set(mine) & set(theirs))
