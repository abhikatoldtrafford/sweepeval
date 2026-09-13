"""Context retention scorer (spec §11.5).

Scored from **final outputs only** — the tool never inspects intermediate
state, because it has none to inspect.

Recall is per unit-run and depth-tagged, so the AUC falls out of aggregation:
the depth lives on the Observation, and the bootstrap stratifies on it (§13.3).
That stratification is not optional. Resampling twelve conversations
unstratified empties a depth in a few percent of replicates, and the trapezoid
is undefined there.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from sweepeval.capabilities.detect import Capability
from sweepeval.capabilities.normalise import normalise
from sweepeval.schema.call import Call
from sweepeval.schema.metric import MetricSpec
from sweepeval.schema.observation import Observation, Verdict
from sweepeval.schema.unit import Unit
from sweepeval.scorers.base import ScoreContext, register

# The trapezoid arithmetic lives in `stats`: the ranker needs the same
# statistic this scorer's metric is reported with, and importing it through
# `scorers` would drag the request layer into `rank` (the layering contract
# refuses that). Re-exported here so the scorer's public surface is unchanged.
from sweepeval.stats.retention import (
    depth_at_floor,
    retention_auc,
    retention_curve,
)

__all__ = ["ContextScorer", "depth_at_floor", "retention_auc", "retention_curve"]


@dataclass
class ContextScorer:
    family: str = "context"
    version: int = 1
    requires: frozenset[Capability] = field(
        default_factory=lambda: frozenset({Capability.MULTI_TURN})
    )

    def metrics(self) -> list[MetricSpec]:
        return [
            MetricSpec(
                metric="fact_recall", family="context", direction="maximize",
                unit="rate", cluster_key="conversation",
            ),
            MetricSpec(
                metric="context_retention_auc", family="context",
                direction="maximize", unit="auc", cluster_key="conversation",
            ),
        ]

    def score(
        self, unit: Unit, calls: Sequence[Call], context: ScoreContext
    ) -> list[Observation]:
        expected = next(
            (s.expect for s in unit.scoring if s.kind == "fact_recall" and s.expect),
            None,
        )
        if not expected:
            return [
                context.observation(
                    scorer=self.family, version=self.version, metric="fact_recall",
                    family=self.family, verdict=Verdict.SKIPPED,
                    reason="template declares no expected fact", unit=unit,
                )
            ]

        answer = normalise(context.text)
        if not answer:
            return [
                context.observation(
                    scorer=self.family, version=self.version, metric="fact_recall",
                    family=self.family, verdict=Verdict.UNSCORABLE,
                    reason="no text extracted from the final turn", unit=unit,
                )
            ]

        recalled = normalise(expected) in answer
        return [
            context.observation(
                scorer=self.family, version=self.version, metric="fact_recall",
                family=self.family,
                verdict=Verdict.PASS if recalled else Verdict.FAIL,
                value=1.0 if recalled else 0.0,
                reason=f"depth {unit.depth}: {'recalled' if recalled else 'lost'}",
                unit=unit,
            )
        ]





register(ContextScorer())

