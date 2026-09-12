"""Monte Carlo coverage for I2 (spec §13.5). **This simulation gates M6.**

I2 claims domination is asserted only when a paired test rejects at the stated
family-wise error rate. That is a claim about a *rate*, and the only way to
check a rate is to measure it.

An earlier revision's property test asserted the code matched its own
definition of domination, which cannot fail for the reason the invariant
actually breaks. This generates configs with known ground truth and counts how
often the pipeline gets it wrong.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from sweepeval.rank.domination import compare_all, frontier_of
from sweepeval.schema.objective import Objective
from sweepeval.stats.paired import paired_difference

pytestmark = pytest.mark.simulation

ALPHA = 0.05
N_CLUSTERS = 24
N_TRIALS = 120


def _objective(name: str, min_effect: float = 0.02) -> Objective:
    return Objective(
        id=name,
        display_label=name,
        direction="maximize",
        family="test",
        cluster_key="probe",
        min_effect=min_effect,
        min_effect_kind="absolute",
        weighting="test",
    )


def _run_trial(
    true_rates: dict[str, dict[str, float]],
    objectives: Sequence[Objective],
    rng: np.random.Generator,
    n_clusters: int = N_CLUSTERS,
) -> tuple[tuple[str, ...], object]:
    """Sample per-cluster outcomes and run the full domination pipeline."""
    ids = [f"p{i}" for i in range(n_clusters)]
    config_ids = sorted(true_rates)

    # Per (config, objective, cluster) Bernoulli draws, on the SHARED cluster
    # set — which is what I4 guarantees and what the pairing relies on.
    observed: dict[tuple[str, str], dict[str, float]] = {}
    for config in config_ids:
        for objective in objectives:
            rate = true_rates[config][objective.id]
            draws = rng.binomial(1, rate, size=n_clusters).astype(float)
            observed[(config, objective.id)] = dict(zip(ids, draws, strict=True))

    def paired(a: str, b: str, objective: Objective) -> object:
        va = observed[(a, objective.id)]
        vb = observed[(b, objective.id)]

        def stat(values: dict[str, float]):
            def inner(drawn: Sequence[str]) -> float:
                return sum(values[c] for c in drawn) / len(drawn)

            return inner

        return paired_difference(
            ids, stat(va), stat(vb),
            direction=objective.direction,
            margin=objective.min_effect,
            seed=int(rng.integers(0, 2**31)),
            n_resamples=300, alpha=ALPHA, bounded=True,
        )

    result = compare_all(config_ids, objectives, paired, alpha=ALPHA)  # type: ignore[arg-type]
    frontier, _ = frontier_of(config_ids, result)
    return frontier, result


def test_false_domination_rate_is_at_or_below_alpha() -> None:
    """Configs that are genuinely identical must almost never dominate.

    This is the number I2 is a claim about.
    """
    objectives = [_objective(f"m{i}") for i in range(3)]
    rates = {c: {o.id: 0.80 for o in objectives} for c in ("c1", "c2", "c3")}

    rng = np.random.default_rng(11)
    false_claims = 0
    for _ in range(N_TRIALS):
        _, result = _run_trial(rates, objectives, rng)
        if any(v.dominates for v in result.verdicts.values()):  # type: ignore[attr-defined]
            false_claims += 1

    rate = false_claims / N_TRIALS
    assert rate <= ALPHA + 0.03, f"false domination rate {rate:.3f} exceeds alpha {ALPHA}"


def test_a_genuinely_dominated_config_is_detected() -> None:
    """The test must also have power, or a rate of zero would pass trivially."""
    objectives = [_objective(f"m{i}") for i in range(3)]
    rates = {
        "strong": {o.id: 0.95 for o in objectives},
        "weak": {o.id: 0.40 for o in objectives},
    }

    rng = np.random.default_rng(3)
    detected = 0
    for _ in range(N_TRIALS):
        _, result = _run_trial(rates, objectives, rng)
        if result.dominates("strong", "weak"):  # type: ignore[attr-defined]
            detected += 1

    assert detected / N_TRIALS > 0.8, "a large true difference should be found"


def test_a_config_better_on_one_and_worse_on_another_is_never_dominated() -> None:
    """The defining Pareto case. Domination here would be a correctness bug,
    not a statistical one."""
    fast = _objective("speed")
    safe = _objective("safety")
    rates = {
        "fast_unsafe": {"speed": 0.95, "safety": 0.40},
        "slow_safe": {"speed": 0.40, "safety": 0.95},
    }

    rng = np.random.default_rng(5)
    for _ in range(40):
        _, result = _run_trial(rates, [fast, safe], rng)
        assert not result.dominates("fast_unsafe", "slow_safe")  # type: ignore[attr-defined]
        assert not result.dominates("slow_safe", "fast_unsafe")  # type: ignore[attr-defined]


def test_a_trade_off_config_always_stays_on_the_frontier() -> None:
    """I2 restated as the property users actually rely on."""
    objectives = [_objective("speed"), _objective("safety")]
    rates = {
        "fast_unsafe": {"speed": 0.95, "safety": 0.40},
        "slow_safe": {"speed": 0.40, "safety": 0.95},
        "middling": {"speed": 0.70, "safety": 0.70},
    }

    rng = np.random.default_rng(7)
    for _ in range(40):
        frontier, _ = _run_trial(rates, objectives, rng)
        assert "fast_unsafe" in frontier
        assert "slow_safe" in frontier


def test_a_difference_below_min_effect_does_not_dominate() -> None:
    """§13.6: statistical significance is not importance.

    A one-point advantage measured over enough clusters is significant and
    still meaningless.
    """
    objectives = [_objective("m0", min_effect=0.10)]
    rates = {"a": {"m0": 0.80}, "b": {"m0": 0.83}}

    rng = np.random.default_rng(13)
    claimed = 0
    for _ in range(60):
        _, result = _run_trial(rates, objectives, rng, n_clusters=200)
        if result.dominates("b", "a"):  # type: ignore[attr-defined]
            claimed += 1

    assert claimed == 0, "a 3-point gap must not dominate against a 10-point margin"


def test_more_configs_do_not_inflate_the_false_domination_rate() -> None:
    """What the family-wise correction is for.

    Twelve configs is 132 ordered pairs. Uncorrected, a 5% per-pair error rate
    would make a false claim near-certain in every sweep.
    """
    objectives = [_objective("m0"), _objective("m1")]
    rates = {f"c{i}": {"m0": 0.80, "m1": 0.80} for i in range(8)}

    rng = np.random.default_rng(17)
    false_claims = 0
    trials = 40
    for _ in range(trials):
        _, result = _run_trial(rates, objectives, rng)
        if any(v.dominates for v in result.verdicts.values()):  # type: ignore[attr-defined]
            false_claims += 1

    assert false_claims / trials <= ALPHA + 0.05
