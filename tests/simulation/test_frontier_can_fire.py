"""The domination relation has to be able to fire (spec §13.5, §14, I2).

An audit found it could not. Two independent causes, both of which left "every
config is statistically tied" as the output of a broken discriminator rather
than a finding about the target:

1. **A noisy objective blocked everything.** Domination is an
   intersection-union test, so it needs non-inferiority on *every* objective.
   The latency objective was a p95 over a corpus-sized cluster list with a
   margin scaled off the mean -- non-inferiority was establishable on two
   *identical* distributions only 8% of the time. A config failing every
   security probe stayed on the frontier.
2. **The bootstrap was coarser than the threshold it was compared against.**
   2000 replicates give a resolution of 5e-4; at the 12-config cap the
   tightest Holm threshold is 0.05/132 = 3.8e-4. Holm is step-down, so unless
   some p-value came out exactly 0.0, nothing anywhere was rejected.

The existing simulation missed both because its scenarios move every objective
together and never exercise a tied-but-noisy one.
"""

from __future__ import annotations

import random

import pytest

from sweepeval.rank.frontier import rank_configs
from sweepeval.schema.metric import CIMethod, Estimand, MetricValue
from sweepeval.schema.objective import REGISTRY
from sweepeval.stats.paired import resamples_for
from sweepeval.stats.resample import N_RESAMPLES

UNITS = [f"u{i}" for i in range(24)]
OBJECTIVES = [REGISTRY.get("security_pass_rate"), REGISTRY.get("latency_mean_ms")]


def _mv(point: float) -> MetricValue:
    return MetricValue(
        point=point,
        lo=max(0.0, point * 0.9),
        hi=point * 1.1 if point else 0.1,
        method=CIMethod.cluster_bootstrap,
        n_clusters=len(UNITS),
        alpha=0.05,
        estimand=Estimand.generalization,
    )


def _config(rng: random.Random, security: float, latency_scale: float = 1.0):
    sec = {u: float(rng.random() < security) for u in UNITS}
    lat = {u: rng.lognormvariate(7.0, 0.6) * latency_scale for u in UNITS}
    clusters = {"security_pass_rate": sec, "latency_ms": lat}
    metrics = {
        "security_pass_rate": _mv(sum(sec.values()) / len(UNITS)),
        "latency_mean_ms": _mv(sum(lat.values()) / len(UNITS)),
    }
    return clusters, metrics


# --- the blocker itself ---------------------------------------------------


def test_a_config_that_fails_every_security_probe_is_dominated() -> None:
    """The audit's reproduction. Latency drawn from the same distribution for
    both, so it must not decide anything."""
    rng = random.Random(7)
    clusters, metrics = {}, {}
    for name, rate in (("good", 1.0), ("bad", 0.0)):
        c, m = _config(rng, rate)
        clusters[name], metrics[name] = c, m

    result = rank_configs(
        ["good", "bad"], metrics, clusters, OBJECTIVES, constraints=(), seed=7
    )
    assert "bad" in result.dominated, result.domination.verdicts[
        ("bad", "good")
    ].reason
    assert result.frontier == ("good",)


def test_a_noisy_objective_does_not_block_a_decisive_one() -> None:
    """Non-inferiority on latency has to be establishable, or the IUT can
    never conclude anything."""
    rng = random.Random(11)
    clusters, metrics = {}, {}
    for name, rate in (("good", 0.95), ("bad", 0.30)):
        c, m = _config(rng, rate)
        clusters[name], metrics[name] = c, m

    result = rank_configs(
        ["good", "bad"], metrics, clusters, OBJECTIVES, constraints=(), seed=11
    )
    verdict = result.domination.verdicts[("bad", "good")]
    latency = next(
        c for c in verdict.per_objective if c.objective == "latency_mean_ms"
    )
    assert latency.p_non_inferior < 0.05, latency


# --- I2 still holds: conservative in the unsafe direction -----------------


def test_identical_configs_are_almost_never_falsely_dominated() -> None:
    """I2. Unblocking the relation must not have made it credulous."""
    false_dominations = 0
    trials = 120
    for seed in range(trials):
        rng = random.Random(seed)
        clusters, metrics = {}, {}
        for name in ("a", "b", "c"):
            c, m = _config(rng, 0.8)
            clusters[name], metrics[name] = c, m
        result = rank_configs(
            list(clusters), metrics, clusters, OBJECTIVES, constraints=(), seed=seed
        )
        false_dominations += bool(result.dominated)

    rate = false_dominations / trials
    assert rate <= 0.05, f"false domination {rate:.3f} exceeds the family-wise rate"


def test_a_genuine_difference_is_still_found() -> None:
    """Conservative must not mean blind."""
    found = 0
    trials = 40
    for seed in range(trials):
        rng = random.Random(500 + seed)
        clusters, metrics = {}, {}
        for name, rate in (("good", 0.95), ("bad", 0.45)):
            c, m = _config(rng, rate)
            clusters[name], metrics[name] = c, m
        result = rank_configs(
            list(clusters), metrics, clusters, OBJECTIVES, constraints=(), seed=seed
        )
        found += "bad" in result.dominated
    assert found / trials >= 0.5, f"power {found / trials:.2f} is too low to be useful"


# --- resolution has to clear the threshold --------------------------------


@pytest.mark.parametrize("configs", [3, 6, 10, 12])
def test_the_bootstrap_resolves_finer_than_the_holm_threshold(configs: int) -> None:
    """Otherwise the family cannot reject at all, whatever the data says."""
    alpha = 0.05
    pairs = configs * (configs - 1)
    objectives = 6
    n_resamples = resamples_for(pairs * objectives, alpha)

    # With the (r+1)/(B+1) estimator the smallest achievable p-value is
    # 1/(B+1), and the superiority half is Bonferroni-multiplied by the
    # objective count.
    smallest_reportable = objectives / (n_resamples + 1)
    tightest_threshold = alpha / pairs
    assert smallest_reportable < tightest_threshold, (
        f"{configs} configs: min p {smallest_reportable:.2e} cannot clear "
        f"threshold {tightest_threshold:.2e}"
    )


def test_a_small_family_does_not_pay_for_a_large_one() -> None:
    assert resamples_for(6) == N_RESAMPLES


def test_a_p_value_is_never_reported_as_exactly_zero() -> None:
    """A bootstrap cannot demonstrate p=0, and reporting it as such is what
    made the whole family hinge on one replicate."""
    from sweepeval.stats.paired import paired_difference

    a = dict.fromkeys(UNITS, 0.0)
    b = dict.fromkeys(UNITS, 1.0)
    result = paired_difference(
        UNITS, lambda d: 0.0, lambda d: 0.0, margin=0.02, seed=1,
        values_a=a, values_b=b, n_resamples=2000,
    )
    assert result.p_superior > 0.0
    assert result.p_superior <= 1 / 2001 + 1e-12


# --- an unmeasurable objective must not veto the pair ---------------------


def test_an_objective_with_no_shared_clusters_is_excluded_not_failed() -> None:
    """Found on real data. Two models' guardrail probes were unscorable in
    different places, so they shared no cluster; the IUT takes the maximum
    p-value, that objective returned 1.0, and one unmeasurable dimension
    vetoed every domination in a ten-model sweep.

    §14.5 already excludes a metric for coverage divergence. No shared
    clusters at all is the limiting case.
    """
    rng = random.Random(3)
    clusters, metrics = {}, {}
    for name, rate in (("good", 1.0), ("bad", 0.0)):
        c, m = _config(rng, rate)
        clusters[name], metrics[name] = c, m

    # A guardrail objective whose clusters do not overlap at all.
    clusters["good"]["guardrail_pass_rate"] = {"g1": 1.0, "g2": 1.0}
    clusters["bad"]["guardrail_pass_rate"] = {"g8": 1.0, "g9": 1.0}
    for name in ("good", "bad"):
        metrics[name]["guardrail_pass_rate"] = _mv(1.0)

    objectives = [*OBJECTIVES, REGISTRY.get("guardrail_pass_rate")]
    result = rank_configs(
        ["good", "bad"], metrics, clusters, objectives, constraints=(), seed=3
    )
    verdict = result.domination.verdicts[("bad", "good")]
    assert "guardrail_pass_rate" in verdict.incomparable
    assert "bad" in result.dominated, verdict.reason


def test_a_pair_sharing_nothing_at_all_does_not_dominate() -> None:
    """Excluding objectives must not become excluding all of them."""
    rng = random.Random(5)
    clusters, metrics = {}, {}
    for name, prefix in (("good", "x"), ("bad", "y")):
        c, m = _config(rng, 1.0 if name == "good" else 0.0)
        clusters[name] = {
            metric: {f"{prefix}{k}": v for k, v in table.items()}
            for metric, table in c.items()
        }
        metrics[name] = m
    result = rank_configs(
        ["good", "bad"], metrics, clusters, OBJECTIVES, constraints=(), seed=5
    )
    assert not result.dominated


def test_a_tie_without_power_is_not_reported_as_being_worse() -> None:
    """"worse on at least one objective" when the objectives were exactly tied
    is the absence-of-evidence conflation the spec refuses elsewhere."""
    rng = random.Random(9)
    clusters, metrics = {}, {}
    for name in ("a", "b"):
        c, m = _config(rng, 0.8)
        clusters[name], metrics[name] = c, m
    result = rank_configs(
        ["a", "b"], metrics, clusters, OBJECTIVES, constraints=(), seed=9
    )
    for verdict in result.domination.verdicts.values():
        if verdict.dominates:
            continue
        if not any(
            c.comparable and c.difference <= -c.applied_margin
            for c in verdict.per_objective
        ):
            assert "worse on" not in verdict.reason, verdict.reason
