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

        margin = overrides.get(objective.id, objective.min_effect)
        if objective.min_effect_kind == "relative":
            scale = abs(sum(base[c] for c in shared) / len(shared)) or 1.0
            margin *= scale

        # b = baseline, a = current, so a positive difference means the
        # BASELINE is ahead — that is, the current run got worse.
        result = paired_difference(
            shared,
            _stat(current),
            _stat(base),
            direction=objective.direction,
            margin=margin,
            alpha=alpha,
            seed=seed,
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

        diffs.append(
            MetricDiff(
                metric=metric,
                baseline_point=sum(base[c] for c in shared) / len(shared),
                current_point=sum(current[c] for c in shared) / len(shared),
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
    return GateVerdict(
        ok=ok,
        exit_code=ExitCode.PASS if ok else ExitCode.REGRESSION,
        regressions=regressions,
        diffs=tuple(diffs),
        hard_fails=tuple(hard_fails),
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
