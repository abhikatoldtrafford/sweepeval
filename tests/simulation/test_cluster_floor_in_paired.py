"""The cluster floor binds the p-value, not just the interval (§13.4, I2).

§13.4: "Fewer than 8 clusters never bootstraps." `stats/resample.py` honours
that and falls back to a t-interval. `stats/paired.py` did not: it
bootstrapped anyway, widened the *interval* with an Agresti-Coull pad, and
left `p_superior` / `p_non_inferior` computed from the forbidden bootstrap --
and those are what Holm consumes.

At three clusters the replicate distribution is degenerate about a tenth of
the time, so the p-value floored at 1/(B+1) = 0.0005 while the interval
printed beside it straddled zero. An audit measured the family-wise
false-domination rate at 8.5% [6.2, 11.6] against I2's stated 5% -- the
confidence interval clears alpha, so it was a violation and not noise.

The replacement is an exact sign-flip permutation test: at most 2**7 = 128
assignments, enumerable outright, no seed. It is self-limiting in the right
way -- the smallest p it can produce is 1/2**n, so at three clusters no
arrangement of the data reaches alpha=0.05. Not a refusal; three paired
observations simply do not contain that much evidence.
"""

from __future__ import annotations

import random

import pytest

from sweepeval.rank.frontier import rank_configs
from sweepeval.schema.metric import CIMethod, Estimand, MetricValue
from sweepeval.schema.objective import REGISTRY
from sweepeval.stats.paired import paired_difference
from sweepeval.stats.resample import CLUSTER_FLOOR

OBJECTIVES = [REGISTRY.get("security_pass_rate"), REGISTRY.get("latency_mean_ms")]


def _mv(point: float, n: int) -> MetricValue:
    return MetricValue(
        point=point, lo=max(0.0, point * 0.9), hi=point * 1.1 if point else 0.1,
        method=CIMethod.cluster_bootstrap, n_clusters=n, alpha=0.05,
        estimand=Estimand.generalization,
    )


def _sweep(seed: int, n_clusters: int, rates: dict[str, float]):
    rng = random.Random(seed)
    units = [f"u{i}" for i in range(n_clusters)]
    clusters, metrics = {}, {}
    for name, rate in rates.items():
        sec = {u: float(rng.random() < rate) for u in units}
        lat = {u: rng.lognormvariate(7.0, 0.6) for u in units}
        clusters[name] = {"security_pass_rate": sec, "latency_ms": lat}
        metrics[name] = {
            "security_pass_rate": _mv(sum(sec.values()) / n_clusters, n_clusters),
            "latency_mean_ms": _mv(sum(lat.values()) / n_clusters, n_clusters),
        }
    return rank_configs(
        list(clusters), metrics, clusters, OBJECTIVES, constraints=(), seed=seed
    )


# --- I2 below the floor ---------------------------------------------------


@pytest.mark.parametrize("n_clusters", [3, 5, 7])
def test_identical_configs_are_not_falsely_dominated_below_the_floor(
    n_clusters: int,
) -> None:
    """The audit's measurement, at the sizes where it failed. Four configs
    drawn from one distribution: any domination is a false one."""
    trials = 200
    false_dominations = sum(
        1
        for seed in range(trials)
        if _sweep(seed, n_clusters, {f"c{k}": 0.8 for k in range(4)}).dominated
    )
    rate = false_dominations / trials
    assert rate <= 0.05, f"{n_clusters} clusters: false domination {rate:.3f}"


# --- the mechanism --------------------------------------------------------


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7])
def test_below_the_floor_the_test_is_exact_not_a_bootstrap(n: int) -> None:
    ids = [f"c{i}" for i in range(n)]
    a = dict.fromkeys(ids, 0.0)
    b = dict.fromkeys(ids, 1.0)
    result = paired_difference(
        ids, lambda d: 0.0, lambda d: 0.0, margin=0.02, seed=1,
        values_a=a, values_b=b,
    )
    assert result.method is CIMethod.permutation
    assert result.evidence is not None
    assert result.evidence["smallest_possible_p"] == pytest.approx(1 / 2**n)
    assert result.p_superior >= 1 / 2**n


def test_at_three_clusters_nothing_can_reach_significance() -> None:
    """Maximally separated data -- every cluster favours b. The exact test
    still cannot clear 0.05, because 1/8 is the smallest p three paired
    observations can produce. The bootstrap reported 0.0005 here."""
    ids = ["c0", "c1", "c2"]
    result = paired_difference(
        ids, lambda d: 0.0, lambda d: 0.0, margin=0.02, seed=1,
        values_a=dict.fromkeys(ids, 0.0), values_b=dict.fromkeys(ids, 1.0),
    )
    assert result.p_superior >= 0.125


def test_the_decision_and_the_printed_interval_agree() -> None:
    """The concrete symptom: `p_superior = 0.000500` beside an interval
    containing zero. A reader checking the interval and a reader checking the
    verdict got opposite answers from the same object."""
    for n in range(3, CLUSTER_FLOOR):
        ids = [f"c{i}" for i in range(n)]
        result = paired_difference(
            ids, lambda d: 0.0, lambda d: 0.0, margin=0.0, seed=1,
            values_a=dict.fromkeys(ids, 0.0), values_b=dict.fromkeys(ids, 1.0),
        )
        straddles_zero = result.lo <= 0 <= result.hi
        assert not (straddles_zero and result.p_superior < 0.05), (
            f"{n} clusters: p={result.p_superior} with CI "
            f"[{result.lo}, {result.hi}]"
        )


def test_a_statistic_with_no_per_cluster_difference_declines() -> None:
    """The retention curve is not separable per cluster, so there is nothing
    to sign-flip. Declining beats permuting something that is not the
    statistic."""
    ids = [f"c{i}" for i in range(5)]
    result = paired_difference(
        ids, lambda d: 0.5, lambda d: 0.9, margin=0.02, seed=1,
    )
    assert result.p_superior == 1.0
    assert result.method is CIMethod.none


# --- power above the floor is untouched -----------------------------------


@pytest.mark.parametrize(("n_clusters", "floor"), [(12, 0.4), (24, 0.6)])
def test_a_real_difference_is_still_found_at_the_corpus_sizes(
    n_clusters: int, floor: float
) -> None:
    """Every shipped family carries floor + 2 clusters or more, so this is
    where production runs actually sit. Conservative below the floor must not
    have cost anything above it."""
    trials = 80
    found = sum(
        1
        for seed in range(trials)
        if "bad" in _sweep(500 + seed, n_clusters, {"good": 0.95, "bad": 0.30}).dominated
    )
    assert found / trials >= floor, f"{n_clusters} clusters: power {found / trials:.2f}"
