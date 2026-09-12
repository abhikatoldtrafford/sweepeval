"""The paired comparison primitive (spec §13.3).

One method, used for every objective.

The cluster is the probe (for context retention, the conversation). I4
guarantees both configs faced the identical Units with identical canaries, so
the probe is a **matched block** and the difference can be taken within it.
That pairing is where the power comes from at N=3, and it was available for
free — an earlier revision of the spec compared two one-sample intervals and
threw it away, which made the domination test effectively alpha≈0.003 and meant
nothing would ever be dominated.

Each objective resamples **its own** cluster set, not one global set shared
across objectives. Per-objective resampling loses nothing, because §13.5's
combination rule is valid under arbitrary dependence between objectives.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from sweepeval.schema.metric import CIMethod, Estimand, Flag, MetricValue
from sweepeval.stats.resample import (
    CLUSTER_FLOOR,
    MIN_CLUSTERS_FOR_ANY_INTERVAL,
    N_RESAMPLES,
    agresti_coull_interval,
    resample_indices,
)

__all__ = ["PairedResult", "paired_difference"]

Statistic = Callable[[Sequence[str]], float]


@dataclass(frozen=True)
class PairedResult:
    """The difference ``b - a`` with its interval and one-sided tail areas."""

    difference: float
    """``b - a``, sign-oriented so positive always means b is ahead."""

    lo: float
    hi: float
    n_clusters: int

    p_superior: float
    """P-value for H0: b is NOT ahead by at least ``margin``.

    Small means b is meaningfully better.
    """

    p_non_inferior: float
    """P-value for H0: b is BEHIND by more than ``margin``.

    Small means b is meaningfully not-worse. This is a real test, not a failure
    to reject: establishing non-inferiority by "we did not detect a
    difference" is absence of evidence, and §9.1 refuses that reasoning
    elsewhere for the same reason.
    """

    method: CIMethod
    margin: float = 0.0
    """The margin actually applied, after a relative min_effect was scaled.

    Reported rather than recomputed by the caller: a relative min_effect of
    0.10 becomes some number of milliseconds, and printing the 0.10 beside a
    latency difference in milliseconds tells the reader nothing true.
    """

    flags: tuple[Flag, ...] = ()
    evidence: Mapping[str, Any] | None = None

    def as_metric(self, estimand: Estimand, alpha: float) -> MetricValue:
        return MetricValue(
            point=self.difference,
            lo=self.lo,
            hi=self.hi,
            method=self.method,
            n_clusters=self.n_clusters,
            alpha=alpha,
            estimand=estimand,
            flags=self.flags,
        )


def paired_difference(
    cluster_ids: Sequence[str],
    statistic_a: Statistic,
    statistic_b: Statistic,
    *,
    direction: str = "maximize",
    margin: float = 0.0,
    n_resamples: int = N_RESAMPLES,
    alpha: float = 0.05,
    seed: int = 0,
    strata: Mapping[str, str] | None = None,
    bounded: bool = False,
) -> PairedResult:
    """Paired cluster bootstrap of ``statistic_b - statistic_a``.

    Both statistics receive the **same** resampled cluster set on every
    replicate. That is what cancels probe-difficulty variance: a probe that is
    hard for one config is hard for the other, and the difference does not care
    how hard it was.

    ``direction`` orients the difference so positive always means b is ahead,
    and callers never have to remember whether higher is better for a metric.

    Both p-values are computed here rather than derived from the interval by
    the caller. They are tail areas of the replicate distribution against
    shifted thresholds, and that distribution only exists inside this function
    — reconstructing them from ``lo``/``hi`` outside it is how the
    non-inferiority half comes to have inverted semantics.
    """
    n = len(cluster_ids)
    sign = 1.0 if direction == "maximize" else -1.0
    observed = sign * (statistic_b(list(cluster_ids)) - statistic_a(list(cluster_ids)))

    if n < MIN_CLUSTERS_FOR_ANY_INTERVAL:
        # Below three clusters there is no honest interval and therefore no
        # comparison. Returning a wide one would still license a domination
        # claim; returning p=1 refuses to.
        return PairedResult(
            difference=observed,
            lo=observed,
            hi=observed,
            n_clusters=n,
            p_superior=1.0,
            p_non_inferior=1.0,
            method=CIMethod.none,
            margin=margin,
            flags=(Flag.NO_VALID_INTERVAL, Flag.LOW_N),
            evidence={"reason": "fewer than 3 clusters"},
        )

    rng = np.random.default_rng(seed)
    replicates = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        drawn = resample_indices(list(cluster_ids), rng, strata)
        replicates[i] = sign * (statistic_b(drawn) - statistic_a(drawn))

    lo = float(np.percentile(replicates, 100 * alpha / 2))
    hi = float(np.percentile(replicates, 100 * (1 - alpha / 2)))
    lo, hi = min(lo, observed), max(hi, observed)

    flags: list[Flag] = []
    if n < CLUSTER_FLOOR:
        flags.append(Flag.LOW_N)
        # Below the floor the bootstrap under-covers. Widen by the
        # Agresti-Coull halfwidth of a bounded statistic at this cluster
        # count, which is the same correction §13.4 applies to single-config
        # intervals and for the same reason.
        if bounded:
            ac_lo, ac_hi = agresti_coull_interval(0.5, n, alpha)
            pad = (ac_hi - ac_lo) / 2
            lo, hi = min(lo, observed - pad), max(hi, observed + pad)

    # Both hypotheses are tail areas of the same distribution against
    # thresholds shifted by the margin (§13.5, §13.6).
    #
    #   superiority     H0: difference <= +margin  -> P(replicate <= +margin)
    #   non-inferiority H0: difference <= -margin  -> P(replicate <= -margin)
    #
    # Non-inferiority is therefore a rejection, not a failure to detect.
    p_superior = float(np.mean(replicates <= margin))
    p_non_inferior = float(np.mean(replicates <= -margin))

    return PairedResult(
        difference=observed,
        lo=lo,
        hi=hi,
        n_clusters=n,
        p_superior=p_superior,
        p_non_inferior=p_non_inferior,
        method=CIMethod.cluster_bootstrap,
        margin=margin,
        flags=tuple(flags),
        evidence={
            "resamples": n_resamples,
            "stratified": strata is not None,
            "margin": margin,
        },
    )
