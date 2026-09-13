"""Baseline diff and the gate rule (spec §16, D26).

Gate flapping is the primary failure mode of tools in this category, and the
rule here is the explicit design against it.

An earlier revision fired when "the new point falls outside the baseline
interval AND the new interval excludes the baseline point". With equal standard
errors that is exactly ``diff > 1.96 * SE`` where ``sd(diff) = 1.414 * SE``,
i.e. **z = 1.386, one-sided p ≈ 0.083** — roughly three times flappier than a
correct paired test at alpha=0.05, marketed as the anti-flapping design. With six
objectives that is a ~39% chance of a false fire per gate run.

Rev 2 replaces it with a paired one-sided test against a **minimum practical
effect**, corrected by Holm across gated metrics. Statistical significance
alone never fails a build.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from sweepeval.schema.objective import Objective
from sweepeval.stats.multiplicity import holm, holm_thresholds
from sweepeval.stats.paired import PairedResult, paired_difference
from sweepeval.stats.statistic import (
    CURVE_OBJECTIVES,
    statistic_for,
    usable_strata,
)

__all__ = [
    "DEFAULT_GATE_ALPHA",
    "ExitCode",
    "GateVerdict",
    "MetricDiff",
    "gate_metrics",
]

DEFAULT_GATE_ALPHA = 0.05


class ExitCode(int, Enum):
    """§16's exit codes."""

    PASS = 0
    REGRESSION = 1
    COMPARABILITY_REFUSED = 2
    USAGE_ERROR = 3


@dataclass(frozen=True)
class MetricDiff:
    metric: str
    baseline_point: float
    current_point: float
    difference: float
    """Oriented so negative always means worse, whatever the direction."""

    min_effect: float
    p_value: float
    adjusted_threshold: float | None
    regressed: bool
    reason: str

    @property
    def improved(self) -> bool:
        return self.difference >= self.min_effect and not self.regressed


@dataclass
class GateVerdict:
    ok: bool
    exit_code: ExitCode
    regressions: tuple[MetricDiff, ...] = ()
    diffs: tuple[MetricDiff, ...] = ()
    hard_fails: tuple[str, ...] = ()
    refusals: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    not_gated: tuple[str, ...] = ()
    """Requested objectives that could not be tested at all.

    A PARTIAL miss used to be discarded: `no_data` was read only when *every*
    metric was missing, so a gate asked for five metrics, able to test one,
    exited 0 with no annotation, no JSON field and nothing on stdout. Four of
    five lost to unscorability is routine -- guardrail coverage in the shipped
    scorecard run ranged 0/20 to 13/20 -- so this is the common case, not the
    edge one.
    """

    degraded: tuple[str, ...] = ()
    """Objectives gated on a weaker statistic than they are reported with.

    In practice: `context_retention_auc` against a baseline written before
    `Baseline.strata` existed, which carries no depth labels, so the gate
    compares mean recall and the report prints the depth-weighted area.
    """

    @property
    def incomplete(self) -> bool:
        return bool(self.not_gated or self.degraded)

    def explain(self) -> str:
        lines: list[str] = []
        for diff in self.diffs:
            mark = "REGRESSION" if diff.regressed else ("improved" if diff.improved else "ok")
            lines.append(
                f"{diff.metric:34s} {diff.baseline_point:>9.4g} -> "
                f"{diff.current_point:>9.4g}  {mark}  ({diff.reason})"
            )
        for hard in self.hard_fails:
            lines.append(f"HARD FAIL  {hard}")
        for refusal in self.refusals:
            lines.append(f"REFUSED    {refusal}")
        for name in self.not_gated:
            lines.append(f"NOT GATED  {name}: no cluster shared with the baseline")
        for name in self.degraded:
            lines.append(
                f"DEGRADED   {name}: gated on the mean, not the statistic it is "
                "reported with (the baseline carries no strata)"
            )
        lines.extend(self.notes)
        lines.append(f"exit {self.exit_code.value}")
        return "\n".join(lines)


def gate_metrics(
    objectives: Sequence[Objective],
    baseline_clusters: Mapping[str, Mapping[str, float]],
    current_clusters: Mapping[str, Mapping[str, float]],
    *,
    hard_fails: Sequence[str] = (),
    alpha: float = DEFAULT_GATE_ALPHA,
    seed: int = 0,
    min_effect_overrides: Mapping[str, float] | None = None,
    gate_on: Sequence[str] | None = None,
    strata: Mapping[str, Mapping[str, str]] | None = None,
) -> GateVerdict:
    """Compare a run against a baseline, per metric.

    ``*_clusters`` map metric id to ``{cluster_id: value}`` — the per-cluster
    values, not summary points. The paired test needs them: the whole reason
    the gate is not flappy is that the same clusters appear on both sides.
    """
    gated = set(gate_on) if gate_on is not None else {o.id for o in objectives}
    overrides = dict(min_effect_overrides or {})

    # A name that resolves to no objective gates nothing, silently. A typo in
    # --gate-on used to exit 0 on a run whose security rate had gone from 100%
    # to 0%, which is the worst possible way for a gate to fail.
    unknown = gated - {o.id for o in objectives}
    if unknown:
        raise ValueError(
            f"--gate-on names {', '.join(sorted(unknown))}, which is not a known "
            f"objective. Available: {', '.join(sorted(o.id for o in objectives))}"
        )

    raw: dict[str, float] = {}
    detail: dict[str, tuple[Objective, PairedResult, float]] = {}
    no_data: list[str] = []
    degraded: list[str] = []

    for objective in objectives:
        if objective.id not in gated:
            continue
        # Objective ids are not always cluster-table keys: latency_p95_ms is
        # stored as latency_ms and cost_per_probe as tokens_out. Looking up by
        # id found nothing for four of the six objectives, so opting in to any
        # of them gated nothing at all and said nothing about it.
        metric = objective.metric
        base = baseline_clusters.get(metric, {})
        current = current_clusters.get(metric, {})
        shared = sorted(set(base) & set(current))
        if not shared:
            no_data.append(objective.id)
            continue

        # The statistic the objective is REPORTED with, from the same dispatch
        # the frontier uses. This was an unweighted mean for every objective,
        # so the gate compared a different quantity from the one it named:
        # `context_retention_auc` -- gated by default -- went 0.6458 to 0.3542
        # as a depth-weighted AUC and 0.5000 to 0.5000 as a mean, and exited 0.
        metric_strata = (strata or {}).get(metric) or {}
        spec_current = statistic_for(objective, current, metric_strata)
        spec_base = statistic_for(objective, base, metric_strata)
        if objective.id in CURVE_OBJECTIVES and not usable_strata(
            metric_strata, current
        ):
            # Gating on the mean is still better than not gating, but saying
            # so is not optional: a baseline written before `Baseline.strata`
            # existed carries no depth labels, and the number the gate then
            # compares is not the one the report prints.
            degraded.append(objective.id)

        margin = overrides.get(objective.id, objective.min_effect)
        if objective.min_effect_kind == "relative":
            scale = abs(spec_base.fn(shared)) or 1.0
            margin *= scale

        # b = baseline, a = current, so a positive difference means the
        # BASELINE is ahead — that is, the current run got worse.
        result = paired_difference(
            shared,
            spec_current.fn,
            spec_base.fn,
            direction=objective.direction,
            margin=margin,
            alpha=alpha,
            seed=seed,
            strata=metric_strata or None,
        )
        raw[objective.id] = result.p_superior
        detail[objective.id] = (objective, result, margin)

    rejected = holm(raw, alpha=alpha)
    thresholds = holm_thresholds(raw, alpha=alpha)

    diffs: list[MetricDiff] = []
    for metric, (objective, result, margin) in sorted(detail.items()):
        key = objective.metric
        base = baseline_clusters[key]
        current = current_clusters[key]
        shared = sorted(set(base) & set(current))
        regressed = rejected.get(metric, False)

        # The printed points come from the same statistic as the test. They
        # were unweighted means, so a gate row could read "0.5 -> 0.5 ok" for
        # an AUC that had gone 0.65 -> 0.35: not only the wrong verdict but
        # two numbers that made it look justified.
        metric_strata = (strata or {}).get(key) or {}
        point_base = statistic_for(objective, base, metric_strata).fn
        point_current = statistic_for(objective, current, metric_strata).fn

        diffs.append(
            MetricDiff(
                metric=metric,
                baseline_point=point_base(shared),
                current_point=point_current(shared),
                # Flip so negative always reads as worse.
                difference=-result.difference,
                min_effect=margin,
                p_value=result.p_superior,
                adjusted_threshold=thresholds.get(metric),
                regressed=regressed,
                reason=(
                    f"worse by at least {margin:.4g}, p={result.p_superior:.4g} "
                    f"<= {thresholds.get(metric, alpha):.4g}"
                    if regressed
                    else _why_not_regressed(result, margin, objective)
                ),
            )
        )

    regressions = tuple(d for d in diffs if d.regressed)

    # A gate that tested nothing is not a pass. Exiting 0 because every
    # requested metric was missing tells the user their build is clean when in
    # fact nothing was checked.
    if not diffs:
        return GateVerdict(
            ok=False,
            exit_code=ExitCode.COMPARABILITY_REFUSED,
            regressions=(),
            diffs=(),
            hard_fails=tuple(hard_fails),
            notes=(
                "no metric could be gated: "
                + (
                    f"{', '.join(no_data)} had no clusters shared with the baseline"
                    if no_data
                    else "no objective was selected"
                ),
            ),
        )

    ok = not regressions and not hard_fails
    # `not_gated` does not by itself fail the build: a family going unscorable
    # is a fact about the run, not evidence of a regression, and exiting 1 on
    # it would teach people to pass --gate-on to silence the gate. It is
    # reported everywhere instead -- terminal, JSON and a GHA warning -- so a
    # green build that tested one metric of five cannot look like a green
    # build that tested five.
    return GateVerdict(
        ok=ok,
        exit_code=ExitCode.PASS if ok else ExitCode.REGRESSION,
        regressions=regressions,
        diffs=tuple(diffs),
        hard_fails=tuple(hard_fails),
        not_gated=tuple(sorted(no_data)),
        degraded=tuple(sorted(degraded)),
    )


def _stat(values: Mapping[str, float]) -> Callable[[Sequence[str]], float]:
    def inner(drawn: Sequence[str]) -> float:
        return sum(values[c] for c in drawn) / len(drawn)

    return inner


def _why_not_regressed(
    result: PairedResult, margin: float, objective: Objective
) -> str:
    if result.difference < margin:
        return f"movement below the {margin:.4g} practical-effect margin"
    return "movement not significant after the family-wise correction"
