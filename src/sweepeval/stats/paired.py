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

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from sweepeval.schema.metric import CIMethod, Estimand, Flag, MetricValue
from sweepeval.stats.resample import (
    CLUSTER_FLOOR,
    MAX_RESAMPLES,
    MIN_CLUSTERS_FOR_ANY_INTERVAL,
    N_RESAMPLES,
    agresti_coull_interval,
    resample_index_matrix,
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


def resamples_for(n_hypotheses: int, alpha: float = 0.05) -> int:
    """Replicates needed for a p-value fine enough to clear a Holm threshold.

    With the (r+1)/(B+1) estimator the smallest achievable p-value is
    1/(B+1), so to reject at the tightest threshold ``alpha/n_hypotheses``
    the bootstrap needs ``B > n_hypotheses/alpha``. Below that the resolution
    of the test is coarser than the threshold it is compared against, and the
    whole family fails to reject regardless of the data.

    Doubling that gives headroom so a single replicate on the wrong side of
    the margin does not decide the frontier.
    """
    needed = math.ceil(2 * n_hypotheses / alpha)
    return max(N_RESAMPLES, min(needed, MAX_RESAMPLES))


def _vectorised_replicates(
    cluster_ids: Sequence[str],
    values_a: Mapping[str, float],
    values_b: Mapping[str, float],
    sign: float,
    n_resamples: int,
    rng: np.random.Generator,
    strata: Mapping[str, str] | None,
    quantile: float | None,
) -> npt.NDArray[np.float64]:
    """All replicates at once, so a larger B is affordable."""
    a = np.array([values_a[c] for c in cluster_ids], dtype=float)
    b = np.array([values_b[c] for c in cluster_ids], dtype=float)

    codes = None
    if strata is not None:
        labels = sorted({strata[c] for c in cluster_ids})
        index = {label: i for i, label in enumerate(labels)}
        codes = np.array([index[strata[c]] for c in cluster_ids], dtype=np.int64)

    draws = resample_index_matrix(len(cluster_ids), n_resamples, rng, codes)

    if quantile is None:
        # A paired bootstrap of a difference of means is a one-sample
        # bootstrap of the mean of the per-cluster differences.
        differences = b - a
        means: npt.NDArray[np.float64] = differences[draws].mean(axis=1)
        return sign * means

    quantiles: npt.NDArray[np.float64] = np.quantile(
        b[draws], quantile, axis=1, method="nearest"
    ) - np.quantile(a[draws], quantile, axis=1, method="nearest")
    return sign * quantiles


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
    values_a: Mapping[str, float] | None = None,
    values_b: Mapping[str, float] | None = None,
    quantile: float | None = None,
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

    if n < CLUSTER_FLOOR:
        # §13.4: "Fewer than 8 clusters never bootstraps." `resample.py`
        # honours that and falls back to a t-interval; this function did not.
        # It bootstrapped anyway, widened the *interval* with an Agresti-Coull
        # pad, and left `p_superior` computed from the forbidden bootstrap --
        # which is what Holm consumes. At 3 clusters the replicate
        # distribution is degenerate about 10% of the time, so p floored at
        # 1/(B+1) while the interval printed beside it straddled zero, and an
        # audit measured the family-wise false-domination rate at 8.5%
        # [6.2, 11.6] against a stated 5%.
        return _exact_paired(
            list(cluster_ids), statistic_a, statistic_b, sign,
            observed=observed, margin=margin, alpha=alpha,
            values_a=values_a, values_b=values_b, bounded=bounded,
        )

    rng = np.random.default_rng(seed)

    if values_a is not None and values_b is not None:
        replicates = _vectorised_replicates(
            list(cluster_ids), values_a, values_b, sign,
            n_resamples, rng, strata, quantile,
        )
    else:
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
    # The (r+1)/(B+1) convention, not the raw proportion. A bootstrap cannot
    # demonstrate p = 0, and reporting zero is both dishonest and load-bearing
    # here: Holm is step-down, so at the 12-config cap -- where the tightest
    # threshold is below the resolution of the old 2000 replicates -- nothing
    # was rejected at all unless some p came out exactly zero.
    #
    #   superiority     H0: difference <= +margin  -> P(replicate <= +margin)
    #   non-inferiority H0: difference <= -margin  -> P(replicate <= -margin)
    #
    # Non-inferiority is therefore a rejection, not a failure to detect.
    total = replicates.size
    p_superior = float((np.count_nonzero(replicates <= margin) + 1) / (total + 1))
    p_non_inferior = float((np.count_nonzero(replicates <= -margin) + 1) / (total + 1))

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


def _exact_paired(
    cluster_ids: list[str],
    statistic_a: Statistic,
    statistic_b: Statistic,
    sign: float,
    *,
    observed: float,
    margin: float,
    alpha: float,
    values_a: Mapping[str, float] | None,
    values_b: Mapping[str, float] | None,
    bounded: bool,
) -> PairedResult:
    """Below the cluster floor: an exact sign-flip permutation test.

    With at most seven paired clusters the null distribution can be
    enumerated -- 2**n sign assignments, 128 at the largest -- so there is no
    reason to approximate it with a bootstrap the spec forbids at this size.

    It is also self-limiting in the right way. The smallest p-value the test
    can produce is ``1/2**n``: 0.125 at three clusters, 0.031 at five. So at
    three clusters no arrangement of the data reaches alpha=0.05 and no
    domination can be asserted -- not by refusal, but because three paired
    observations genuinely do not contain that much evidence. That is the
    honest version of the p=1 the sub-three branch returns.

    A statistic that is not separable per cluster -- the retention curve --
    has no per-cluster difference to flip, so it declines instead.
    """
    if values_a is None or values_b is None:
        return PairedResult(
            difference=observed, lo=observed, hi=observed, n_clusters=len(cluster_ids),
            p_superior=1.0, p_non_inferior=1.0, method=CIMethod.none, margin=margin,
            flags=(Flag.NO_VALID_INTERVAL, Flag.LOW_N),
            evidence={
                "reason": (
                    f"{len(cluster_ids)} clusters is below the floor of "
                    f"{CLUSTER_FLOOR}, and this statistic has no per-cluster "
                    "difference to permute"
                )
            },
        )

    diffs = np.array(
        [sign * (values_b[c] - values_a[c]) for c in cluster_ids], dtype=float
    )
    n = len(diffs)

    # Every assignment of +/- to the n paired differences, as a (2**n, n)
    # matrix of signs. Exhaustive, so the result does not depend on a seed.
    grid = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    means = (grid * diffs).mean(axis=1)

    # H0 for superiority: the true difference is at most +margin. Shift the
    # observed effect by the margin and ask how much of the null mass sits at
    # or above it. `>=` and not `>`: the observed assignment is one of the
    # 2**n, and excluding it is the p=0 the bootstrap was corrected for.
    p_superior = float(np.mean(means >= observed - margin))
    p_non_inferior = float(np.mean(means >= observed + margin))

    # The interval is the same enumeration's quantiles, so the decision and
    # the printed bounds cannot disagree -- which was the concrete symptom.
    lo = float(np.quantile(means, alpha / 2)) + observed - float(means.mean())
    hi = float(np.quantile(means, 1 - alpha / 2)) + observed - float(means.mean())
    lo, hi = min(lo, observed), max(hi, observed)
    if bounded:
        lo, hi = max(-1.0, lo), min(1.0, hi)

    return PairedResult(
        difference=observed,
        lo=lo,
        hi=hi,
        n_clusters=n,
        p_superior=p_superior,
        p_non_inferior=p_non_inferior,
        method=CIMethod.permutation,
        margin=margin,
        flags=(Flag.LOW_N,),
        evidence={
            "reason": (
                f"{n} clusters is below the floor of {CLUSTER_FLOOR}; exact "
                f"sign-flip test over {2**n} assignments"
            ),
            "smallest_possible_p": 1.0 / 2**n,
        },
    )
