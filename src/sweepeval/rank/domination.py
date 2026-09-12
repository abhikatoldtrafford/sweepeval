"""Domination (spec §13.5, §14). I2 load-bearing.

The only module allowed to decide that one config dominates another — enforced
by the layering contract, because a domination decision made anywhere else
would not go through Holm and the invariant would not hold.

I2: "Domination is asserted only when a paired test rejects at the stated
family-wise error rate. No config is excluded from the frontier on weaker
evidence."
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from sweepeval.schema.objective import Objective
from sweepeval.stats.multiplicity import (
    ObjectiveComparison,
    PairVerdict,
    holm,
    holm_thresholds,
    pair_p_value,
)
from sweepeval.stats.paired import PairedResult

__all__ = ["DominationResult", "compare_all", "comparisons_for_pair"]

PairedFn = Callable[[str, str, Objective], PairedResult]


@dataclass
class DominationResult:
    verdicts: dict[tuple[str, str], PairVerdict] = field(default_factory=dict)
    alpha: float = 0.05

    def dominates(self, b: str, a: str) -> bool:
        """True when ``b`` dominates ``a``."""
        verdict = self.verdicts.get((a, b))
        return bool(verdict and verdict.dominates)

    def dominators_of(self, config_id: str) -> tuple[str, ...]:
        return tuple(
            sorted(v.b for (a, _), v in self.verdicts.items() if a == config_id and v.dominates)
        )

    def tied(self, a: str, b: str) -> bool:
        return not self.dominates(a, b) and not self.dominates(b, a)


def comparisons_for_pair(
    a: str,
    b: str,
    objectives: Sequence[Objective],
    paired: PairedFn,
) -> list[ObjectiveComparison]:
    """Build the per-objective comparisons for one ordered pair.

    Both p-values come from ``stats.paired``, which computes them from the
    replicate distribution against margin-shifted thresholds. They are not
    re-derived here from the interval: doing that is how the non-inferiority
    half came to have inverted semantics, declaring a config that was
    *worst* on an objective to be definitively non-inferior on it.
    """
    out: list[ObjectiveComparison] = []
    for objective in objectives:
        result = paired(a, b, objective)
        out.append(
            ObjectiveComparison(
                objective=objective.id,
                difference=result.difference,
                min_effect=objective.min_effect,
                p_superior=result.p_superior,
                p_non_inferior=result.p_non_inferior,
            )
        )
    return out


def compare_all(
    config_ids: Sequence[str],
    objectives: Sequence[Objective],
    paired: PairedFn,
    *,
    alpha: float = 0.05,
) -> DominationResult:
    """Every ordered pair, combined by §13.5 and corrected by Holm."""
    raw: dict[str, float] = {}
    detail: dict[str, PairVerdict] = {}

    for a in config_ids:
        for b in config_ids:
            if a == b:
                continue
            comparisons = comparisons_for_pair(a, b, objectives, paired)
            p_pair, p_ni, p_sup = pair_p_value(comparisons)
            key = f"{b}>{a}"
            raw[key] = p_pair
            detail[key] = PairVerdict(
                a=a,
                b=b,
                p_pair=p_pair,
                p_non_inferior=p_ni,
                p_superior=p_sup,
                per_objective=tuple(comparisons),
                wins=tuple(
                    c.objective for c in comparisons if c.difference >= c.min_effect
                ),
                concedes=tuple(
                    c.objective for c in comparisons if c.difference <= -c.min_effect
                ),
            )

    rejected = holm(raw, alpha=alpha)
    thresholds = holm_thresholds(raw, alpha=alpha)

    verdicts: dict[tuple[str, str], PairVerdict] = {}
    for key, verdict in detail.items():
        verdict.dominates = rejected.get(key, False)
        verdict.adjusted_threshold = thresholds.get(key)
        verdict.reason = (
            "dominates: non-inferior on every objective and better on at least one"
            if verdict.dominates
            else _why_not(verdict)
        )
        verdicts[(verdict.a, verdict.b)] = verdict

    return DominationResult(verdicts=verdicts, alpha=alpha)


def _why_not(verdict: PairVerdict) -> str:
    if verdict.p_non_inferior > verdict.p_superior:
        return "not dominating: worse on at least one objective"
    if verdict.p_superior >= 1.0:
        return "not dominating: no objective improved by at least its min_effect"
    return "not dominating: evidence does not survive the family-wise correction"


def frontier_of(
    config_ids: Sequence[str], result: DominationResult
) -> tuple[tuple[str, ...], Mapping[str, tuple[str, ...]]]:
    """Non-dominated configs, plus each dominated config's dominators."""
    dominated: dict[str, tuple[str, ...]] = {}
    for config_id in config_ids:
        dominators = result.dominators_of(config_id)
        if dominators:
            dominated[config_id] = dominators
    frontier = tuple(c for c in config_ids if c not in dominated)
    return frontier, dominated


__all__ += ["frontier_of"]
