"""Domination p-values and multiplicity control (spec §13.5). I2 load-bearing.

Pareto domination is "no worse on every objective, strictly better on at least
one". That is a **conjunction of non-inferiority claims** combined with a
**disjunction of superiority claims**, and the two halves need different
treatment.

An earlier revision tested both with a single ``max`` over per-objective
superiority p-values. That statistic is the p-value for *strict superiority on
every objective* — a different and much stronger hypothesis than the rule it
was paired with. It also established "A is not significantly better" by
*failing to reject*, which is absence of evidence: with wide intervals the
non-inferiority half became near-vacuous and domination collapsed into "better
on at least one objective", which is not domination at all.

Holm, not Benjamini-Hochberg. The guarantee wanted is "with probability at
least 1-alpha, nothing was wrongly declared dominated" — family-wise error control.
BH controls the false *discovery* rate and permits a nonzero expected
proportion of false claims, which cannot underwrite an invariant stated as a
bound.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

__all__ = [
    "ObjectiveComparison",
    "PairVerdict",
    "holm",
    "pair_p_value",
]


@dataclass(frozen=True)
class ObjectiveComparison:
    """One objective's paired result, already oriented so higher is better."""

    objective: str
    difference: float
    """``b - a``, positive when b is ahead."""

    min_effect: float
    """As declared on the objective: absolute, or a relative fraction."""

    p_superior: float
    """P(b is not better than a by at least the applied margin)."""

    p_non_inferior: float
    """P(b is worse than a by more than the applied margin)."""

    applied_margin: float = 0.0
    """What ``min_effect`` became in the metric's own units, after scaling.

    Reported separately because a relative min_effect of 0.10 printed beside a
    latency difference in milliseconds tells the reader nothing true.
    """


@dataclass
class PairVerdict:
    a: str
    b: str
    p_pair: float
    p_non_inferior: float
    p_superior: float
    dominates: bool = False
    adjusted_threshold: float | None = None
    per_objective: tuple[ObjectiveComparison, ...] = ()
    reason: str = ""
    wins: tuple[str, ...] = field(default_factory=tuple)
    concedes: tuple[str, ...] = field(default_factory=tuple)


def pair_p_value(comparisons: Sequence[ObjectiveComparison]) -> tuple[float, float, float]:
    """The two-half construction of §13.5.

    Returns ``(p_pair, p_non_inferior, p_superior)``.

    * Non-inferiority is an **intersection-union test**: b must be
      non-inferior on *every* objective, and the max of the per-objective
      p-values is exactly that test. An IUT needs no correction within the
      conjunction — free rigour an earlier revision missed.
    * Superiority is a **union**: b need only be better on one, so the minimum
      p-value is Bonferroni-corrected by the number of objectives.
    * The pair's p-value is the max of the two halves, because both must hold.
    """
    if not comparisons:
        return 1.0, 1.0, 1.0

    p_non_inferior = max(c.p_non_inferior for c in comparisons)
    p_superior = min(1.0, len(comparisons) * min(c.p_superior for c in comparisons))
    return max(p_non_inferior, p_superior), p_non_inferior, p_superior


def holm(p_values: Mapping[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni step-down. Returns ``{key: rejected}``.

    Family-wise error control across the k(k-1) ordered pairs.
    """
    if not p_values:
        return {}

    ordered = sorted(p_values.items(), key=lambda kv: (kv[1], kv[0]))
    total = len(ordered)
    rejected: dict[str, bool] = {}
    still_rejecting = True

    for index, (key, p_value) in enumerate(ordered):
        threshold = alpha / (total - index)
        if still_rejecting and p_value <= threshold:
            rejected[key] = True
        else:
            # Holm is step-down: the first failure stops the procedure, and
            # everything after it is retained regardless of its own p-value.
            still_rejecting = False
            rejected[key] = False

    return rejected


def holm_thresholds(p_values: Mapping[str, float], alpha: float = 0.05) -> dict[str, float]:
    """The threshold each key was compared against, for the report."""
    ordered = sorted(p_values.items(), key=lambda kv: (kv[1], kv[0]))
    total = len(ordered)
    return {key: alpha / (total - index) for index, (key, _) in enumerate(ordered)}


__all__ += ["holm_thresholds"]
