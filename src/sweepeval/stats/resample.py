"""Interval primitives (spec §13.3, §13.4).

Moved into M0 because M3's sampling-effect test needs intervals before the
statistics milestone exists.

**The cluster floor.** Below 8 clusters a percentile bootstrap cannot produce
an interval wider than the observed range, and its coverage sits far below
nominal. It is a *narrow wrong* interval, not a conservatively wide one — which
is why an earlier revision's "bootstrap at n=3 with a LOW_N flag" was the most
dangerous line in its statistics section. Below the floor this module falls
back to a t-interval; below 3 it declines to give one at all. BCa needs n≳20
and is not implemented anywhere in v0.1.

**Stratification.** ``strata`` maps cluster id to a stratum label; when given,
resampling draws within each stratum with replacement, preserving the
per-stratum count. This is mandatory for ``context_retention_auc``: resampling
9 conversations unstratified empties at least one depth in about 7.6% of
replicates ((2/3)⁹ per depth, three depths) and the trapezoid is undefined
there.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from sweepeval.schema.metric import CIMethod, Estimand, Flag, MetricValue

__all__ = [
    "CLUSTER_FLOOR",
    "MIN_CLUSTERS_FOR_ANY_INTERVAL",
    "N_RESAMPLES",
    "agresti_coull_interval",
    "cluster_bootstrap",
    "no_valid_interval",
    "resample_indices",
    "t_interval",
]

CLUSTER_FLOOR = 8
MIN_CLUSTERS_FOR_ANY_INTERVAL = 3
N_RESAMPLES = 2000

ClusterT = TypeVar("ClusterT")

# Two-sided t critical values at alpha=0.05, indexed by degrees of freedom.
# Table rather than scipy: the only alpha v0.1 uses is 0.05, and adding scipy
# for eleven numbers is not a trade worth making.
_T_CRITICAL_95: Mapping[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def _t_critical(df: int, alpha: float) -> float:
    if not math.isclose(alpha, 0.05):
        raise ValueError(
            f"t_interval only tabulates alpha=0.05; got {alpha}. "
            "Add a table entry rather than interpolating."
        )
    return _T_CRITICAL_95.get(df, 1.96)


MAX_RESAMPLES = 40_000
"""Ceiling on the adaptive resample count.

Beyond this the bootstrap costs more than the sweep that produced the data,
and a family needing it is one where the sweep is too large for the evidence
rather than one that needs more replicates.
"""


def resample_index_matrix(
    n: int,
    n_resamples: int,
    rng: np.random.Generator,
    strata_codes: npt.NDArray[np.int64] | None = None,
) -> npt.NDArray[np.int64]:
    """``(n_resamples, n)`` matrix of cluster indices, drawn with replacement.

    The vectorised twin of :func:`resample_indices`. The per-replicate Python
    loop cost roughly a millisecond a replicate, which capped the achievable
    number of resamples at 2000 -- and 2000 gives a p-value resolution of
    5e-4, coarser than the tightest Holm threshold at the 12-config cap
    (0.05/132 = 3.8e-4). At that point nothing could be rejected unless its
    p-value came out exactly zero, so a single replicate on the wrong side of
    the margin took the frontier from eleven dominations to none.

    With ``strata_codes`` the draw is stratified: each replicate takes, from
    each stratum, exactly as many members as that stratum has, so no stratum
    can come back empty and the trapezoid stays defined (§13.3).
    """
    if strata_codes is None:
        return rng.integers(0, n, size=(n_resamples, n))

    columns: list[npt.NDArray[np.int64]] = []
    for code in np.unique(strata_codes):
        members = np.flatnonzero(strata_codes == code)
        columns.append(
            members[rng.integers(0, members.size, size=(n_resamples, members.size))]
        )
    return np.concatenate(columns, axis=1)


def resample_indices(
    cluster_ids: Sequence[ClusterT],
    rng: np.random.Generator,
    strata: Mapping[ClusterT, str] | None = None,
) -> list[ClusterT]:
    """Draw one bootstrap resample of cluster ids, with replacement.

    With ``strata``, draws within each stratum and preserves its size, so no
    stratum can come back empty.
    """
    if strata is None:
        idx = rng.integers(0, len(cluster_ids), size=len(cluster_ids))
        return [cluster_ids[i] for i in idx]

    by_stratum: dict[str, list[ClusterT]] = {}
    for cid in cluster_ids:
        by_stratum.setdefault(strata[cid], []).append(cid)

    drawn: list[ClusterT] = []
    for label in sorted(by_stratum):
        members = by_stratum[label]
        idx = rng.integers(0, len(members), size=len(members))
        drawn.extend(members[i] for i in idx)
    return drawn


def agresti_coull_interval(
    mean: float, n_clusters: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Agresti-Coull interval for a mean of values bounded in [0, 1].

    Exists because both the percentile bootstrap and the t-interval collapse to
    zero width when every cluster agrees, which is common and not rare: at 24
    clusters and a true rate of 0.85, all clusters pass about 2% of the time; at
    8 clusters, 27%; at 5 clusters, 44%. A zero-width interval at 1.0 excludes
    the true rate every single time, and measured coverage without this
    correction was 0.75 at the cluster floor and 0.57 below it, against a
    nominal 0.95.

    Adding two pseudo-successes and two pseudo-failures pulls the estimate off
    the boundary and restores coverage. This is the behaviour spec rev 1 got
    from Wilson intervals for rates and that rev 2 lost when it unified
    everything on the cluster bootstrap.
    """
    if not math.isclose(alpha, 0.05):
        raise ValueError(f"only alpha=0.05 is tabulated; got {alpha}")
    z = 1.96
    n_tilde = n_clusters + z**2
    p_tilde = (mean * n_clusters + z**2 / 2) / n_tilde
    half = z * math.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
    return max(0.0, p_tilde - half), min(1.0, p_tilde + half)


def cluster_bootstrap(
    cluster_ids: Sequence[ClusterT],
    statistic: Callable[[Sequence[ClusterT]], float],
    *,
    estimand: Estimand,
    n_resamples: int = N_RESAMPLES,
    alpha: float = 0.05,
    seed: int = 0,
    strata: Mapping[ClusterT, str] | None = None,
    bounded: bool = False,
    extra_flags: Sequence[Flag] = (),
) -> MetricValue:
    """Percentile cluster bootstrap, with the floor enforced.

    ``statistic`` receives a resampled sequence of cluster ids and returns the
    metric computed over exactly those clusters.

    ``bounded=True`` declares the statistic is a mean of values in [0, 1] — a
    pass rate, a repeatability rate, a normalised AUC. The interval is then
    widened to the union with :func:`agresti_coull_interval`, which is what
    keeps coverage near nominal when clusters agree. Continuous statistics such
    as latency and cost leave it ``False``.
    """
    n = len(cluster_ids)
    point = statistic(list(cluster_ids))

    if n < MIN_CLUSTERS_FOR_ANY_INTERVAL:
        return no_valid_interval(
            point, n, alpha, estimand, extra_flags=(Flag.LOW_N, *extra_flags)
        )

    if n < CLUSTER_FLOOR:
        # Below the floor: a t-interval over the per-cluster values, which is
        # wide and honest, rather than a bootstrap, which would be narrow and
        # wrong.
        values = [statistic([cid]) for cid in cluster_ids]
        return t_interval(
            values,
            alpha=alpha,
            estimand=estimand,
            point=point,
            bounded=bounded,
            extra_flags=(Flag.LOW_N, *extra_flags),
        )

    rng = np.random.default_rng(seed)
    replicates = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        replicates[i] = statistic(resample_indices(cluster_ids, rng, strata))

    lo = float(np.percentile(replicates, 100 * alpha / 2))
    hi = float(np.percentile(replicates, 100 * (1 - alpha / 2)))
    # The percentile interval can exclude the observed point on a skewed
    # statistic. Widening to contain it keeps MetricValue constructible and is
    # the conservative direction.
    lo, hi = min(lo, point), max(hi, point)

    if bounded:
        ac_lo, ac_hi = agresti_coull_interval(point, n, alpha)
        lo, hi = min(lo, ac_lo), max(hi, ac_hi)

    return MetricValue(
        point=point,
        lo=lo,
        hi=hi,
        method=CIMethod.cluster_bootstrap,
        n_clusters=n,
        alpha=alpha,
        estimand=estimand,
        flags=tuple(extra_flags),
    )


def t_interval(
    values: Sequence[float],
    *,
    estimand: Estimand,
    alpha: float = 0.05,
    point: float | None = None,
    bounded: bool = False,
    extra_flags: Sequence[Flag] = (),
) -> MetricValue:
    """Student-t interval over cluster-level values. Wide by design.

    ``bounded=True`` widens to the union with :func:`agresti_coull_interval`,
    for the same reason as in :func:`cluster_bootstrap`: with all clusters in
    agreement the sample standard deviation is zero and the t-interval has zero
    width, which never covers.
    """
    n = len(values)
    if n < MIN_CLUSTERS_FOR_ANY_INTERVAL:
        centre = point if point is not None else (float(np.mean(values)) if n else 0.0)
        return no_valid_interval(
            centre, n, alpha, estimand, extra_flags=(Flag.LOW_N, *extra_flags)
        )

    array = np.asarray(values, dtype=float)
    centre = point if point is not None else float(array.mean())
    stderr = float(array.std(ddof=1) / math.sqrt(n))
    half = _t_critical(n - 1, alpha) * stderr
    lo, hi = centre - half, centre + half

    if bounded:
        ac_lo, ac_hi = agresti_coull_interval(centre, n, alpha)
        lo, hi = min(lo, ac_lo), max(hi, ac_hi)

    return MetricValue(
        point=centre,
        lo=lo,
        hi=hi,
        method=CIMethod.t,
        n_clusters=n,
        alpha=alpha,
        estimand=estimand,
        flags=tuple(extra_flags),
    )


def no_valid_interval(
    point: float,
    n_clusters: int,
    alpha: float,
    estimand: Estimand,
    *,
    extra_flags: Sequence[Flag] = (),
) -> MetricValue:
    """Decline to give an interval, loudly (I3)."""
    flags = {Flag.NO_VALID_INTERVAL, *extra_flags}
    return MetricValue(
        point=point,
        method=CIMethod.none,
        n_clusters=n_clusters,
        alpha=alpha,
        estimand=estimand,
        flags=tuple(sorted(flags, key=lambda f: f.value)),
    )
