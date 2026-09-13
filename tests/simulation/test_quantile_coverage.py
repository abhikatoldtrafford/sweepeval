"""A quantile interval has to cover what it says (spec §13.3, I3).

The bootstrap percentile interval under-covers badly for a quantile. Measured
against a known lognormal, `latency_p95_ms` covered its true value

    0.713 at 24 clusters
    0.883 at 69 clusters

with `alpha=0.05` printed beside it and no flag. §13.3 says an objective
missing nominal by more than three points falls back; D46 applied that to the
default set only, and the p95 is promotable with `--objectives`.

The replacement is the distribution-free order-statistic interval: if B is
Binomial(n, q), the qth quantile lies between the l-th and u-th order
statistics with probability P(l <= B <= u-1), which is exact at any n for any
continuous distribution.

The important part is what it does when it *cannot* answer. A p95's true value
lies above every observed probe with probability 0.95**n -- 29% at 24 clusters
-- so below 72 clusters no interval ending at the sample maximum can cover it,
whatever method produced it. That is a property of the question. It declines,
and §13.3's documented fallback to p90 (which needs 36) is emitted alongside,
under its own name.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sweepeval.execute.aggregation import (
    LATENCY_FALLBACK_OBJECTIVE,
    LATENCY_OBJECTIVE,
)
from sweepeval.schema.metric import CIMethod, Estimand, Flag
from sweepeval.schema.objective import REGISTRY
from sweepeval.stats.resample import _binomial_quantile, order_statistic_interval

MU, SIGMA = 6.0, 0.6


def _true_quantile(q: float) -> float:
    from statistics import NormalDist

    return float(math.exp(MU + SIGMA * NormalDist().inv_cdf(q)))


def _coverage(q: float, n: int, trials: int = 400) -> tuple[float, float]:
    """Returns (declined fraction, coverage among the intervals given)."""
    rng = np.random.default_rng(11)
    true = _true_quantile(q)
    declined = hit = given = 0
    for _ in range(trials):
        values = list(rng.lognormal(MU, SIGMA, n))
        value = order_statistic_interval(
            values, q, estimand=Estimand.generalization
        )
        if Flag.NO_VALID_INTERVAL in value.flags:
            declined += 1
            continue
        given += 1
        assert value.lo is not None and value.hi is not None
        hit += value.lo <= true <= value.hi
    return declined / trials, (hit / given if given else float("nan"))


# --- coverage where an interval is given ----------------------------------


@pytest.mark.parametrize(("q", "n"), [(0.95, 72), (0.95, 120), (0.90, 40), (0.90, 69)])
def test_the_interval_covers_at_least_its_nominal_rate(q: float, n: int) -> None:
    declined, coverage = _coverage(q, n)
    assert declined == 0.0
    assert coverage >= 0.93, f"q={q} n={n}: coverage {coverage:.3f}"


def test_the_bootstrap_it_replaced_did_not() -> None:
    """Not an aspiration. This is the measurement that made the change
    necessary, and it keeps the test above from being vacuous."""
    from sweepeval.stats.resample import cluster_bootstrap

    rng = np.random.default_rng(0)
    true = _true_quantile(0.95)
    hit = 0
    trials = 300
    for k in range(trials):
        values = rng.lognormal(MU, SIGMA, 24)
        table = {f"c{i}": float(v) for i, v in enumerate(values)}

        def statistic(ids, t=table):
            ordered = sorted(t[i] for i in ids)
            return ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]

        value = cluster_bootstrap(
            sorted(table), statistic, seed=k, estimand=Estimand.generalization
        )
        hit += value.lo is not None and value.lo <= true <= value.hi
    assert hit / trials < 0.90, "the bootstrap suddenly covers; recheck the premise"


# --- and declines rather than inventing a bound ---------------------------


@pytest.mark.parametrize("n", [12, 24, 40, 69])
def test_a_p95_below_72_clusters_declines_an_interval(n: int) -> None:
    """Clamping to the sample maximum is where the 0.713 came from. I3 allows
    a metric to decline an interval; it does not allow a wrong one."""
    declined, _ = _coverage(0.95, n, trials=50)
    assert declined == 1.0


@pytest.mark.parametrize(
    ("q", "needed"), [(0.95, 72), (0.90, 36), (0.80, 17), (0.75, 13)]
)
def test_the_sample_size_a_quantile_needs_is_what_the_binomial_says(
    q: float, needed: int
) -> None:
    """The threshold is derived, not tuned. One below and there is no upper
    order statistic to use."""
    assert _binomial_quantile(needed, q, 0.975) + 1 <= needed
    assert _binomial_quantile(needed - 1, q, 0.975) + 1 > needed - 1


def test_a_declined_interval_still_carries_its_point_and_a_flag() -> None:
    value = order_statistic_interval(
        [1.0, 2.0, 3.0] * 8, 0.95, estimand=Estimand.generalization
    )
    assert value.method is CIMethod.none
    assert Flag.NO_VALID_INTERVAL in value.flags
    assert value.point > 0


# --- the fallback is real ---------------------------------------------------


def test_both_tail_quantiles_are_emitted_under_their_own_names() -> None:
    """Reporting a p90 as `latency_p95_ms` is the exact class of defect this
    change exists to remove, so the fallback gets its own id."""
    assert LATENCY_OBJECTIVE == "latency_p95_ms"
    assert LATENCY_FALLBACK_OBJECTIVE == "latency_p90_ms"
    assert REGISTRY.get(LATENCY_FALLBACK_OBJECTIVE).default is False


async def test_a_real_run_reports_the_quantile_it_can_and_declines_the_one_it_cannot(
    tmp_path,
) -> None:
    import tempfile

    from tests.conftest import make_app, make_client

    from sweepeval.execute.evaluate import aevaluate_target

    app = make_app("openai_clean")
    client = make_client(app)
    try:
        result = await aevaluate_target(
            "https://mock.test" + app.scenario.paths[0],
            key="test-key-abcdefgh", client=client, root=tempfile.mkdtemp(),
            runs=2, authorized=True, authorization_prompt=False, seed=7,
        )
    finally:
        await client.aclose()

    p95 = result.metrics[LATENCY_OBJECTIVE]
    p90 = result.metrics[LATENCY_FALLBACK_OBJECTIVE]
    assert p95.n_clusters < 72
    assert Flag.NO_VALID_INTERVAL in p95.flags, "a p95 it cannot bound"
    assert p90.lo is not None and p90.method is CIMethod.order_statistic
