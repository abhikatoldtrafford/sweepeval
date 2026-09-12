"""Dispersion, superiority and equivalence tests (spec §9.1, §13.6).

Lives in ``stats/`` because it computes p-values, and the layering contract
says nothing outside ``stats/`` may. The capability layer owns the *decision
table*; this module owns the arithmetic underneath it.

The equivalence half matters as much as the superiority half. ``INERT`` removes
an axis from the experiment, and a null result is not evidence of no effect —
so that verdict is reachable only through a TOST against a declared margin.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_RESAMPLES",
    "dispersion",
    "dispersion_difference_test",
    "tost_equivalent",
]

DEFAULT_RESAMPLES = 2000


def dispersion(outputs: Sequence[str], similarity: Callable[[str, str], float]) -> float:
    """Mean pairwise dissimilarity over runs. 0.0 when all runs are identical."""
    if len(outputs) < 2:
        return 0.0
    pairs = [1.0 - similarity(a, b) for a, b in combinations(outputs, 2)]
    return float(np.mean(pairs))


def _resampled(
    outputs: Sequence[str],
    similarity: Callable[[str, str], float],
    rng: np.random.Generator,
) -> float:
    """Dispersion of a bootstrap resample **of the runs**.

    Runs, not pairs. The pairwise values are a dependent U-statistic — each run
    appears in n-1 of them — so resampling pairs would treat dependent
    observations as independent and understate the variance, making the test
    over-confident in exactly the direction that removes a live axis.
    """
    idx = rng.integers(0, len(outputs), size=len(outputs))
    return dispersion([outputs[i] for i in idx], similarity)


def _replicates(
    low: Sequence[str],
    high: Sequence[str],
    similarity: Callable[[str, str], float],
    n_resamples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array(
        [
            _resampled(high, similarity, rng) - _resampled(low, similarity, rng)
            for _ in range(n_resamples)
        ]
    )


def dispersion_difference_test(
    low: Sequence[str],
    high: Sequence[str],
    similarity: Callable[[str, str], float],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> tuple[float, dict[str, Any]]:
    """One-sided test that ``high`` disperses more. Returns ``(p, evidence)``."""
    replicates = _replicates(low, high, similarity, n_resamples, seed)
    p_value = float(np.mean(replicates <= 0.0))
    return p_value, {
        "dispersion_low": round(dispersion(low, similarity), 4),
        "dispersion_high": round(dispersion(high, similarity), 4),
        "observed_difference": round(
            dispersion(high, similarity) - dispersion(low, similarity), 4
        ),
    }


def tost_equivalent(
    low: Sequence[str],
    high: Sequence[str],
    similarity: Callable[[str, str], float],
    *,
    margin: float,
    alpha: float = 0.05,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> tuple[bool, dict[str, Any]]:
    """Two one-sided tests for equivalence within ``±margin``."""
    replicates = _replicates(low, high, similarity, n_resamples, seed)
    lo = float(np.percentile(replicates, 100 * alpha))
    hi = float(np.percentile(replicates, 100 * (1 - alpha)))
    equivalent = -margin < lo and hi < margin
    return equivalent, {
        "ci_90": [round(lo, 4), round(hi, 4)],
        "margin": margin,
    }
