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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise

from sweepeval.capabilities.detect import Capability
from sweepeval.capabilities.normalise import normalise
from sweepeval.schema.call import Call
from sweepeval.schema.metric import MetricSpec
from sweepeval.schema.observation import Observation, Verdict
from sweepeval.schema.unit import Unit
from sweepeval.scorers.base import ScoreContext, register

__all__ = ["ContextScorer", "retention_auc", "retention_curve"]


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


def retention_curve(observations: Sequence[Observation]) -> dict[int, float]:
    """Mean recall at each measured depth, in depth order."""
    buckets: dict[int, list[float]] = {}
    for observation in observations:
        if observation.metric != "fact_recall" or observation.depth is None:
            continue
        if observation.verdict in (Verdict.UNSCORABLE, Verdict.SKIPPED):
            continue
        value = observation.value
        if value is None:
            value = 1.0 if observation.verdict is Verdict.PASS else 0.0
        buckets.setdefault(observation.depth, []).append(value)

    return {d: sum(v) / len(v) for d, v in sorted(buckets.items())}


def retention_auc(curve: Mapping[int, float]) -> tuple[float, dict[int, float]]:
    """Normalised trapezoid area over the depth ladder (§11.5).

    Returns ``(auc, weights)``. The weights are published because depth
    *spacing* sets them: on the 3/8/15 ladder the middle depth carries roughly
    half the area purely from geometry, and ``deep`` adds depth 30 and changes
    them — which is precisely why ``profile`` is a hard comparability key.

    A single measured depth has no area, so the AUC degrades to that depth's
    recall and the weight table says so.
    """
    depths = sorted(curve)
    if not depths:
        return 0.0, {}
    if len(depths) == 1:
        return curve[depths[0]], {depths[0]: 1.0}

    span = depths[-1] - depths[0]
    if span <= 0:
        return curve[depths[0]], {depths[0]: 1.0}

    weights: dict[int, float] = {d: 0.0 for d in depths}
    for left, right in pairwise(depths):
        share = (right - left) / (2 * span)
        weights[left] += share
        weights[right] += share

    auc = sum(curve[d] * weights[d] for d in depths)
    return auc, weights


def depth_at_floor(curve: Mapping[int, float], floor: float = 0.5) -> int | None:
    """The first depth where recall crosses below ``floor`` (§11.5)."""
    for depth in sorted(curve):
        if curve[depth] < floor:
            return depth
    return None


register(ContextScorer())

__all__ += ["depth_at_floor"]
