"""``frontier.json`` (spec §6.2, §14.6, D43).

Every comparison, with its p-value and its Holm threshold. That completeness
is the point: a reader who disagrees with a domination verdict has to be able
to see the evidence for it rather than take it on trust, and a reader who
wonders why an obvious winner did not dominate has to be able to find the
objective that blocked it.

No scalar rank field appears anywhere in this document, and a contract test
asserts it. A rank column is a composite score with the arithmetic hidden in
the sort (I1).
"""

from __future__ import annotations

from typing import Any

__all__ = ["FORBIDDEN_FIELDS", "frontier_payload"]

FORBIDDEN_FIELDS = ("rank", "score", "composite", "overall", "total_score")
"""Names a scalar ranking would arrive under. Asserted absent by a test."""


def frontier_payload(result: Any, labels: dict[str, str] | None = None) -> dict[str, Any]:
    """Serialise a ``rank.FrontierResult``."""
    labels = labels or {}

    return {
        "alpha": result.alpha,
        "objectives": [
            {
                "id": o.id,
                "direction": o.direction,
                "family": o.family,
                "cluster_key": o.cluster_key,
                "min_effect": o.min_effect,
                "min_effect_kind": o.min_effect_kind,
                # I1: the within-family weighting is published, not implied.
                "weighting": o.weighting,
                "note": o.note,
            }
            for o in result.objectives
        ],
        "frontier": list(result.frontier),
        "dominated": {k: list(v) for k, v in sorted(result.dominated.items())},
        "clusters": [
            {
                "members": list(c.members),
                "spread": {k: list(v) for k, v in sorted(c.spread.items())},
                "wide": list(c.wide),
            }
            for c in result.clusters
        ],
        "labels": dict(sorted(labels.items())),
        "constraints": _constraints(result),
        "comparisons": _comparisons(result),
        "coverage_blocked": [
            {"a": a, "b": b, "families": list(families)}
            for (a, b), families in sorted(result.blocked_pairs.items())
        ],
        "excluded_metrics": {
            k: list(v) for k, v in sorted(result.excluded_metrics.items())
        },
        "correlation": _correlation(result),
        "points": {k: dict(sorted(v.items())) for k, v in sorted(result.points.items())},
    }


def _constraints(result: Any) -> dict[str, Any]:
    report = result.constraints
    if report is None:
        return {}
    return {
        "eligible": list(report.eligible),
        "violations": [
            {
                "config_id": v.config_id,
                "constraint": v.constraint.describe(),
                "kind": v.constraint.kind,
                "observed": v.point,
                "favourable_bound": v.bound,
            }
            for v in report.violations
        ],
        "not_measured": [
            {"config_id": c, "metric": m} for c, m in report.not_measured
        ],
    }


def _comparisons(result: Any) -> list[dict[str, Any]]:
    """Every ordered pair, whether or not it dominated (D43)."""
    if result.domination is None:
        return []
    out = []
    for (a, b), verdict in sorted(result.domination.verdicts.items()):
        out.append(
            {
                "a": a,
                "b": b,
                "dominates": verdict.dominates,
                "p_pair": verdict.p_pair,
                "p_non_inferior": verdict.p_non_inferior,
                "p_superior": verdict.p_superior,
                "holm_threshold": verdict.adjusted_threshold,
                "reason": verdict.reason,
                "wins": list(verdict.wins),
                "concedes": list(verdict.concedes),
                "per_objective": [
                    {
                        "objective": c.objective,
                        "difference": c.difference,
                        "min_effect": c.min_effect,
                        "applied_margin": c.applied_margin,
                        "p_superior": c.p_superior,
                        "p_non_inferior": c.p_non_inferior,
                    }
                    for c in verdict.per_objective
                ],
            }
        )
    return out


def _correlation(result: Any) -> dict[str, Any]:
    matrix = result.correlation
    if not matrix.computed:
        return {"computed": False, "reason": matrix.reason}
    return {
        "computed": True,
        "objectives": list(matrix.objectives),
        "values": {f"{a}|{b}": v for (a, b), v in sorted(matrix.values.items())},
        "high_pairs": [
            {"a": a, "b": b, "r": r} for a, b, r in matrix.high_pairs()
        ],
    }
