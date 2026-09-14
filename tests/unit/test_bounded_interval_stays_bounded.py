"""A rate's interval cannot leave [0, 1] (§6.4, I3).

`bounded=True` declares the statistic is a mean of values in [0, 1], so the
estimand is a rate. Below the cluster floor the interval is a Student-t over
the per-cluster values, which is unbounded, and the Agresti-Coull step takes
the *union* with it -- widening, never clipping. So a real baseline snapshotted
against a live endpoint recorded this, and wrote it into a file the tool tells
users to commit:

    guardrail_pass_rate   0.667 [-0.768, 2.101]   method=t, n_clusters=3

A pass rate of -0.77 is not a wide interval, it is a wrong one.

Clipping costs no coverage -- the true value is never in the discarded region,
so a 95% interval stays one. That is also what makes it safe for domination:
narrowing an interval normally risks pruning a config that belonged on the
frontier, which §13.5 forbids outright, but removing an impossible region
cannot change which comparisons a correct analysis would support.
"""

from __future__ import annotations

import pytest

from sweepeval.schema.metric import Estimand, Flag
from sweepeval.stats.resample import cluster_bootstrap, t_interval

GENERALIZATION = Estimand.generalization


# --- the case that was recorded ---------------------------------------------


def test_the_live_baselines_interval_is_inside_the_unit_interval() -> None:
    """Three clusters, two passing: the exact shape that produced
    [-0.768, 2.101]."""
    value = t_interval([1.0, 1.0, 0.0], estimand=GENERALIZATION, bounded=True)
    assert value.point == pytest.approx(2 / 3)
    assert value.lo >= 0.0
    assert value.hi <= 1.0


@pytest.mark.parametrize(
    "values",
    [
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0],
        [0.5, 0.5, 1.0],
    ],
)
def test_no_bounded_interval_escapes(values: list[float]) -> None:
    value = t_interval(values, estimand=GENERALIZATION, bounded=True)
    assert 0.0 <= value.lo <= value.hi <= 1.0, (value.lo, value.hi)


def test_the_bootstrap_path_is_bounded_too() -> None:
    """Above the cluster floor the same union runs, so the same clip must."""
    ids = [f"c{i}" for i in range(12)]
    passing = {cid for cid in ids[:9]}
    value = cluster_bootstrap(
        ids,
        lambda chosen: sum(c in passing for c in chosen) / max(1, len(chosen)),
        estimand=GENERALIZATION,
        bounded=True,
        seed=7,
    )
    assert 0.0 <= value.lo <= value.hi <= 1.0, (value.lo, value.hi)


# --- and what must not change ------------------------------------------------


def test_an_unbounded_metric_is_left_alone() -> None:
    """Latency and token counts are not rates. Clipping them to [0, 1] would
    be catastrophic, so the clamp must be reached only via `bounded`."""
    value = t_interval([1400.0, 1800.0, 900.0], estimand=GENERALIZATION)
    assert value.hi > 1.0
    assert value.point == pytest.approx(1366.666, abs=1e-2)


def test_the_point_is_untouched() -> None:
    """Only the ends move. A clamped interval that also moved the estimate
    would be reporting a different measurement."""
    values = [1.0, 1.0, 0.0]
    clamped = t_interval(values, estimand=GENERALIZATION, bounded=True)
    raw = t_interval(values, estimand=GENERALIZATION, bounded=False)
    assert clamped.point == raw.point


def test_a_zero_variance_interval_still_has_width() -> None:
    """The reason the Agresti-Coull union exists: with every cluster agreeing
    the sample standard deviation is zero and the t-interval has zero width,
    which never covers. The clamp must not undo that."""
    value = t_interval([1.0, 1.0, 1.0], estimand=GENERALIZATION, bounded=True)
    assert value.point == 1.0
    assert value.hi == 1.0
    assert value.lo < 1.0, "a zero-width interval at the boundary never covers"


def test_low_n_is_still_flagged_on_the_real_path() -> None:
    """Clipping makes the number printable, not trustworthy. Three clusters is
    still three clusters, and `cluster_bootstrap` is what attaches the flag
    when it drops below the floor and delegates."""
    ids = ["a", "b", "c"]
    passing = {"a", "b"}
    value = cluster_bootstrap(
        ids,
        lambda chosen: sum(c in passing for c in chosen) / max(1, len(chosen)),
        estimand=GENERALIZATION,
        bounded=True,
        seed=7,
    )
    assert value.method.value == "t", "this should be below the bootstrap floor"
    assert Flag.LOW_N in value.flags
    assert 0.0 <= value.lo <= value.hi <= 1.0


def test_two_clusters_decline_the_interval_rather_than_clamping_one() -> None:
    """Below `MIN_CLUSTERS_FOR_ANY_INTERVAL` the answer is no interval at all
    (I3), not a clamped [0, 1] that looks like a measurement."""
    value = t_interval([1.0, 0.0], estimand=GENERALIZATION, bounded=True)
    assert value.lo is None and value.hi is None
    assert Flag.NO_VALID_INTERVAL in value.flags
