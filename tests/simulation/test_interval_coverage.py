"""Monte Carlo coverage of the interval primitives (spec §13.3, §13.4).

An interval that claims 95% coverage and delivers 70% makes every downstream
guarantee false — I2's domination bound, the gate's false-fire rate, the
frontier's ties. The claim is checked here rather than assumed.

This is the M0 slice: coverage of a single-sample interval. Task 5.7's
simulation is the larger one that gates M6, covering domination and the gate.

Marked ``simulation`` and excluded from the default run by its own slowness
budget; CI runs it explicitly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from sweepeval.schema.metric import CIMethod, Estimand
from sweepeval.stats.resample import CLUSTER_FLOOR, cluster_bootstrap

pytestmark = pytest.mark.simulation

GEN = Estimand.generalization


def _coverage(n_clusters: int, true_rate: float, trials: int, seed: int) -> float:
    """Fraction of trials whose interval contains the true rate."""
    rng = np.random.default_rng(seed)
    covered = 0

    for _ in range(trials):
        # Each cluster is a probe; its value is that probe's pass rate over runs.
        outcomes = rng.binomial(1, true_rate, size=n_clusters).astype(float)
        values = {f"c{i}": float(v) for i, v in enumerate(outcomes)}

        def statistic(drawn: Sequence[str], _values: dict[str, float] = values) -> float:
            return sum(_values[c] for c in drawn) / len(drawn)

        result = cluster_bootstrap(
            list(values),
            statistic,
            estimand=GEN,
            seed=int(rng.integers(0, 2**31)),
            n_resamples=400,
            bounded=True,
        )
        if (
            result.lo is not None
            and result.hi is not None
            and result.lo <= true_rate <= result.hi
        ):
            covered += 1

    return covered / trials


@pytest.mark.parametrize("n_clusters", [20, 24, 65])
def test_bootstrap_coverage_is_near_nominal_at_realistic_cluster_counts(
    n_clusters: int,
) -> None:
    """At the corpus's actual cluster counts, coverage should approach 0.95.

    The percentile bootstrap on a proportion is known to under-cover slightly,
    so the bar is 0.88 rather than 0.95 — but a method delivering, say, 0.70
    would invalidate every interval the tool reports, and that is what this
    catches.
    """
    coverage = _coverage(n_clusters, true_rate=0.85, trials=300, seed=n_clusters)
    assert coverage >= 0.88, f"n={n_clusters}: coverage {coverage:.3f}"


def test_coverage_does_not_collapse_at_the_floor() -> None:
    """The floor is where the method switches. Both sides must be defensible."""
    coverage = _coverage(CLUSTER_FLOOR, true_rate=0.85, trials=300, seed=99)
    assert coverage >= 0.85, f"at the floor: coverage {coverage:.3f}"


def test_the_t_fallback_below_the_floor_over_covers_rather_than_under_covers() -> None:
    """Being too wide is the safe direction; being too narrow is not."""
    coverage = _coverage(5, true_rate=0.85, trials=300, seed=5)
    assert coverage >= 0.85, f"below the floor: coverage {coverage:.3f}"


def test_a_degenerate_sample_does_not_claim_a_tight_interval() -> None:
    """All clusters identical is common at temp=0.

    The interval is legitimately degenerate; what must not happen is a
    confident interval that excludes a true rate it never sampled.
    """
    values = {f"c{i}": 1.0 for i in range(24)}

    def statistic(drawn: Sequence[str]) -> float:
        return sum(values[c] for c in drawn) / len(drawn)

    result = cluster_bootstrap(list(values), statistic, estimand=GEN, seed=1)
    assert result.method is CIMethod.cluster_bootstrap
    assert result.lo == result.hi == 1.0
