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

from pydantic import BaseModel, ConfigDict, field_serializer

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

    hard_fails: tuple[str, ...] = ()

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
