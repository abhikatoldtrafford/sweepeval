"""Interval primitives (spec §13.3, §13.4)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pytest

from sweepeval.schema.metric import CIMethod, Estimand, Flag
from sweepeval.stats.resample import (
    CLUSTER_FLOOR,
    agresti_coull_interval,
    cluster_bootstrap,
    no_valid_interval,
    resample_indices,
    t_interval,
)

GEN = Estimand.generalization


def _mean_of(values: dict[str, float]) -> Callable[[Sequence[str]], float]:
    def statistic(drawn: Sequence[str]) -> float:
        return sum(values[c] for c in drawn) / len(drawn)

    return statistic


def _uniform(n: int, value: float = 0.5) -> dict[str, float]:
    return {f"c{i}": value for i in range(n)}


# --- the floor ------------------------------------------------------------


def test_cluster_floor_is_eight() -> None:
    assert CLUSTER_FLOOR == 8


def test_bootstrap_is_used_at_or_above_the_floor() -> None:
    values = {f"c{i}": float(i) for i in range(CLUSTER_FLOOR)}
    result = cluster_bootstrap(list(values), _mean_of(values), estimand=GEN, seed=1)
    assert result.method is CIMethod.cluster_bootstrap
    assert Flag.LOW_N not in result.flags


def test_below_the_floor_falls_back_to_a_t_interval_with_low_n() -> None:
    """§13.4: a bootstrap at n<8 is a narrow wrong interval, not a wide one."""
    values = {f"c{i}": float(i) for i in range(5)}
    result = cluster_bootstrap(list(values), _mean_of(values), estimand=GEN, seed=1)
    assert result.method is CIMethod.t
    assert Flag.LOW_N in result.flags


def test_below_three_clusters_no_interval_is_given() -> None:
    values = {"c0": 1.0, "c1": 2.0}
    result = cluster_bootstrap(list(values), _mean_of(values), estimand=GEN, seed=1)
    assert result.method is CIMethod.none
    assert Flag.NO_VALID_INTERVAL in result.flags
    assert result.lo is None


def test_the_t_fallback_is_wider_than_a_bootstrap_would_have_been() -> None:
    """The whole point of the fallback: honesty about how little we know."""
    values = {f"c{i}": float(i) for i in range(5)}
    fallback = cluster_bootstrap(list(values), _mean_of(values), estimand=GEN, seed=1)

    rng = np.random.default_rng(1)
    ids = list(values)
    replicates = [
        _mean_of(values)(resample_indices(ids, rng)) for _ in range(2000)
    ]
    boot_width = float(np.percentile(replicates, 97.5) - np.percentile(replicates, 2.5))

    assert fallback.lo is not None and fallback.hi is not None
    assert (fallback.hi - fallback.lo) > boot_width


# --- correctness ----------------------------------------------------------


def test_a_constant_statistic_gives_a_degenerate_interval() -> None:
    values = _uniform(10, 0.5)
    result = cluster_bootstrap(list(values), _mean_of(values), estimand=GEN, seed=1)
    assert result.point == 0.5
    assert result.lo == result.hi == 0.5


def test_the_interval_brackets_the_point() -> None:
    values = {f"c{i}": float(i % 3) for i in range(30)}
    result = cluster_bootstrap(list(values), _mean_of(values), estimand=GEN, seed=7)
    assert result.lo is not None and result.hi is not None
    assert result.lo <= result.point <= result.hi


def test_more_clusters_give_a_tighter_interval() -> None:
    def width(n: int) -> float:
        values = {f"c{i}": float(i % 2) for i in range(n)}
        result = cluster_bootstrap(list(values), _mean_of(values), estimand=GEN, seed=3)
        assert result.lo is not None and result.hi is not None
        return result.hi - result.lo

    assert width(200) < width(20)


def test_results_are_reproducible_for_a_fixed_seed() -> None:
    values = {f"c{i}": float(i % 5) for i in range(20)}
    a = cluster_bootstrap(list(values), _mean_of(values), estimand=GEN, seed=42)
    b = cluster_bootstrap(list(values), _mean_of(values), estimand=GEN, seed=42)
    assert (a.lo, a.hi) == (b.lo, b.hi)


def test_different_seeds_give_different_intervals() -> None:
    values = {f"c{i}": float(i % 5) for i in range(20)}
    a = cluster_bootstrap(list(values), _mean_of(values), estimand=GEN, seed=1)
    b = cluster_bootstrap(list(values), _mean_of(values), estimand=GEN, seed=2)
    assert (a.lo, a.hi) != (b.lo, b.hi)


def test_the_estimand_is_recorded() -> None:
    values = _uniform(10)
    assert (
        cluster_bootstrap(
            list(values), _mean_of(values), estimand=Estimand.conditional, seed=1
        ).estimand
        is Estimand.conditional
    )


# --- stratification (§13.3, mandatory for the AUC) ------------------------


def test_unstratified_resampling_can_empty_a_stratum() -> None:
    """Demonstrates why stratification is mandatory, not optional.

    Nine conversations, three per depth. Unstratified, some replicate contains
    no conversation at all from one depth, and the trapezoid is undefined.
    """
    ids = [f"c{i}" for i in range(9)]
    depths = {cid: f"d{i // 3}" for i, cid in enumerate(ids)}

    rng = np.random.default_rng(0)
    empty = 0
    for _ in range(2000):
        drawn = resample_indices(ids, rng)
        if len({depths[c] for c in drawn}) < 3:
            empty += 1
    assert empty > 0


def test_stratified_resampling_never_empties_a_stratum() -> None:
    ids = [f"c{i}" for i in range(9)]
    depths = {cid: f"d{i // 3}" for i, cid in enumerate(ids)}

    rng = np.random.default_rng(0)
    for _ in range(2000):
        drawn = resample_indices(ids, rng, strata=depths)
        assert len({depths[c] for c in drawn}) == 3


def test_stratified_resampling_preserves_per_stratum_counts() -> None:
    ids = [f"c{i}" for i in range(9)]
    depths = {cid: f"d{i // 3}" for i, cid in enumerate(ids)}

    rng = np.random.default_rng(0)
    drawn = resample_indices(ids, rng, strata=depths)
    assert len(drawn) == 9
    for label in {"d0", "d1", "d2"}:
        assert sum(1 for c in drawn if depths[c] == label) == 3


def test_cluster_bootstrap_accepts_strata() -> None:
    ids = [f"c{i}" for i in range(9)]
    depths = {cid: f"d{i // 3}" for i, cid in enumerate(ids)}
    values = {cid: float(i) for i, cid in enumerate(ids)}
    result = cluster_bootstrap(
        ids, _mean_of(values), estimand=GEN, seed=1, strata=depths
    )
    assert result.method is CIMethod.cluster_bootstrap


# --- t_interval and no_valid_interval -------------------------------------


def test_t_interval_widens_as_spread_grows() -> None:
    tight = t_interval([1.0, 1.1, 0.9, 1.0, 1.05], estimand=GEN)
    loose = t_interval([1.0, 5.0, -3.0, 2.0, 0.0], estimand=GEN)
    assert tight.lo is not None and tight.hi is not None
    assert loose.lo is not None and loose.hi is not None
    assert (loose.hi - loose.lo) > (tight.hi - tight.lo)


def test_t_interval_rejects_an_untabulated_alpha() -> None:
    """Better to fail loudly than to interpolate a critical value silently."""
    with pytest.raises(ValueError, match="alpha"):
        t_interval([1.0, 2.0, 3.0, 4.0], estimand=GEN, alpha=0.01)


def test_no_valid_interval_is_flagged_and_bound_free() -> None:
    result = no_valid_interval(0.5, 2, 0.05, GEN)
    assert result.method is CIMethod.none
    assert result.lo is None and result.hi is None
    assert Flag.NO_VALID_INTERVAL in result.flags


# --- boundary correction for bounded statistics (§13.4) -------------------


def test_agresti_coull_covers_when_every_cluster_agrees() -> None:
    """The degenerate case that broke coverage before the correction existed.

    At 24 clusters and a true rate of 0.85, all clusters pass about 2% of the
    time; at 8 clusters, 27%; at 5, 44%. Both the percentile bootstrap and the
    t-interval have zero width there and never cover.
    """
    for n in (5, 8, 24):
        lo, hi = agresti_coull_interval(1.0, n)
        assert lo <= 0.85 <= hi, (n, lo, hi)


def test_agresti_coull_narrows_as_clusters_grow() -> None:
    wide = agresti_coull_interval(0.85, 5)
    tight = agresti_coull_interval(0.85, 65)
    assert (tight[1] - tight[0]) < (wide[1] - wide[0])


def test_agresti_coull_stays_inside_the_unit_interval() -> None:
    for mean in (0.0, 0.5, 1.0):
        lo, hi = agresti_coull_interval(mean, 10)
        assert 0.0 <= lo <= hi <= 1.0


def test_bounded_widens_a_degenerate_bootstrap_interval() -> None:
    values = {f"c{i}": 1.0 for i in range(24)}
    plain = cluster_bootstrap(list(values), _mean_of(values), estimand=GEN, seed=1)
    corrected = cluster_bootstrap(
        list(values), _mean_of(values), estimand=GEN, seed=1, bounded=True
    )
    assert plain.lo == plain.hi == 1.0
    assert corrected.lo is not None and corrected.lo < 1.0
    assert corrected.hi == 1.0


def test_bounded_widens_a_degenerate_t_interval_below_the_floor() -> None:
    values = {f"c{i}": 1.0 for i in range(5)}
    corrected = cluster_bootstrap(
        list(values), _mean_of(values), estimand=GEN, seed=1, bounded=True
    )
    assert corrected.method is CIMethod.t
    assert corrected.lo is not None and corrected.lo < 1.0


def test_bounded_never_narrows_an_interval() -> None:
    """The correction is a union, so it can only widen."""
    values = {f"c{i}": float(i % 4) / 3.0 for i in range(30)}
    plain = cluster_bootstrap(list(values), _mean_of(values), estimand=GEN, seed=2)
    corrected = cluster_bootstrap(
        list(values), _mean_of(values), estimand=GEN, seed=2, bounded=True
    )
    assert plain.lo is not None and corrected.lo is not None
    assert corrected.lo <= plain.lo
    assert corrected.hi is not None and plain.hi is not None
    assert corrected.hi >= plain.hi
