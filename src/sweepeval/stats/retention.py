"""Retention curve arithmetic (spec §11.5, §13.3).

The depth-weighted trapezoid and the resampling statistic built on it. This is
numerics, so it lives in ``stats`` rather than behind the scorer layer -- the
ranker needs the same statistic the aggregator reports, and reaching for it
through ``scorers`` drags the whole request layer along with it. The layering
contract caught exactly that.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import pairwise

from sweepeval.schema.observation import Observation, Verdict

__all__ = [
    "auc_statistic",
    "depth_at_floor",
    "retention_auc",
    "retention_curve",
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


def auc_statistic(
    values: Mapping[str, float], strata: Mapping[str, str]
) -> Callable[[Sequence[str]], float]:
    """The AUC recomputed on a resampled set of conversations.

    The AUC used to borrow ``fact_recall``'s interval and overwrite only the
    point through ``model_copy``, which skips the validator that forbids a
    point outside its own interval. The interval then described mean recall
    per conversation while the number above it was the depth-weighted area --
    so two configs with identical reported AUCs could be declared decisively
    different.
    """

    def statistic(drawn: Sequence[str]) -> float:
        buckets: dict[int, list[float]] = {}
        for cluster in drawn:
            if cluster not in values:
                continue
            label = strata.get(cluster, "")
            if not label.startswith("d"):
                continue
            buckets.setdefault(int(label[1:]), []).append(values[cluster])
        if not buckets:
            return 0.0
        curve = {d: sum(v) / len(v) for d, v in buckets.items()}
        auc, _weights = retention_auc(curve)
        return auc

    return statistic
