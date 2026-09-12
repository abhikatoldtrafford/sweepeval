"""Gate false-fire rate (spec §16, D26). **This simulation gates M6.**

§16 calls flapping "the primary failure mode of tools in this category" and
claims the gate rule is the explicit design against it. That is a claim about a
rate, so it gets measured.

The rule it replaced fired when the new point fell outside the baseline
interval and the new interval excluded the baseline point. With equal standard
errors that is z ≈ 1.386 — a one-sided false-fire rate near 8.3% per metric,
roughly three times flappier than a correct test, and near-certain to fire
somewhere across six objectives. This test would have caught that.
"""

from __future__ import annotations

import numpy as np
import pytest

from sweepeval.schema.objective import Objective
from sweepeval.stats.diff import gate_metrics

pytestmark = pytest.mark.simulation

ALPHA = 0.05
N_CLUSTERS = 24
N_RUNS = 3


def _objective(name: str, min_effect: float = 0.05) -> Objective:
    return Objective(
        id=name, display_label=name, direction="maximize", family="test",
        cluster_key="probe", min_effect=min_effect, min_effect_kind="absolute",
        weighting="test",
    )


def _sample(rate: float, rng: np.random.Generator, n: int = N_CLUSTERS) -> dict[str, float]:
    """Per-cluster pass rates over N runs, as the aggregator produces them."""
    return {
        f"p{i}": float(rng.binomial(N_RUNS, rate) / N_RUNS) for i in range(n)
    }


def _false_fire_rate(
    objectives: list[Objective], true_rate: float, trials: int, seed: int
) -> float:
    """Fraction of trials where an UNCHANGED target fails the gate."""
    rng = np.random.default_rng(seed)
    fired = 0
    for _ in range(trials):
        baseline = {o.id: _sample(true_rate, rng) for o in objectives}
        current = {o.id: _sample(true_rate, rng) for o in objectives}
        verdict = gate_metrics(
            objectives, baseline, current,
            alpha=ALPHA, seed=int(rng.integers(0, 2**31)),
        )
        if not verdict.ok:
            fired += 1
    return fired / trials


def test_an_unchanged_target_almost_never_fails_the_gate() -> None:
    """The anti-flapping claim, measured on one metric."""
    rate = _false_fire_rate([_objective("security_pass_rate")], 0.88, 200, seed=21)
    assert rate <= ALPHA, f"false-fire rate {rate:.3f} exceeds alpha {ALPHA}"


def test_six_objectives_do_not_multiply_the_false_fire_rate() -> None:
    """What Holm across gated metrics is for.

    Uncorrected at 5% each, six objectives would fire somewhere in ~26% of
    runs. A gate that cries wolf once a month gets disabled.
    """
    objectives = [_objective(f"m{i}") for i in range(6)]
    rate = _false_fire_rate(objectives, 0.88, 200, seed=23)
    assert rate <= ALPHA, f"six-objective false-fire rate {rate:.3f}"


def test_a_noisy_target_still_does_not_flap() -> None:
    """Mid-range rates carry the most run-to-run variance."""
    rate = _false_fire_rate([_objective("guardrail_pass_rate")], 0.50, 200, seed=29)
    assert rate <= ALPHA, f"false-fire rate at p=0.50 was {rate:.3f}"


def test_the_gate_still_catches_a_real_regression() -> None:
    """A gate that never fires is not anti-flapping, it is broken."""
    objective = _objective("security_pass_rate")
    rng = np.random.default_rng(31)
    caught = 0
    trials = 100
    for _ in range(trials):
        baseline = {objective.id: _sample(0.92, rng)}
        current = {objective.id: _sample(0.60, rng)}
        verdict = gate_metrics(
            [objective], baseline, current, alpha=ALPHA,
            seed=int(rng.integers(0, 2**31)),
        )
        if not verdict.ok:
            caught += 1
    assert caught / trials > 0.9, f"only caught {caught / trials:.2f} of real regressions"


def test_a_regression_just_over_the_margin_is_caught_more_often_than_not() -> None:
    """Sensitivity at the boundary the margin defines."""
    objective = _objective("security_pass_rate", min_effect=0.05)
    rng = np.random.default_rng(37)
    caught = 0
    trials = 100
    for _ in range(trials):
        baseline = {objective.id: _sample(0.90, rng, n=60)}
        current = {objective.id: _sample(0.72, rng, n=60)}
        verdict = gate_metrics(
            [objective], baseline, current, alpha=ALPHA,
            seed=int(rng.integers(0, 2**31)),
        )
        if not verdict.ok:
            caught += 1
    assert caught / trials > 0.5, f"an 18-point drop was caught only {caught / trials:.2f}"
