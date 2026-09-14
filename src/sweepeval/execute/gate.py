"""Baseline snapshot and the CI gate (spec §16).

``snapshot`` turns a result into a committable baseline. ``gate`` compares an
existing result against one and is a **pure function** — it sends nothing.
``run_gate`` re-runs the target first. Two names because they are two
operations; an earlier revision gave both meanings to one.

Comparability is checked before any statistics run. A refusal exits 2, not 1:
"these results are not comparable" is a different fact from "this got worse",
and reporting the first as the second is how a gate teaches people to ignore it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sweepeval.execute.evaluate import EvaluationResult
from sweepeval.schema.baseline import Baseline
from sweepeval.schema.comparability import compare_keys
from sweepeval.schema.objective import REGISTRY, Objective
from sweepeval.stats.diff import DEFAULT_GATE_ALPHA, ExitCode, GateVerdict, gate_metrics

__all__ = [
    "DEFAULT_GATE_ON",
    "gate",
    "load_baseline",
    "profile_refusal",
    "save_baseline",
    "snapshot",
]


def profile_refusal(profile: str) -> str | None:
    """Why this profile cannot gate, or ``None`` if it can (§16).

    Split out so the answer is available **before** anything is spent. It used
    to live inside :func:`gate_result`, which runs after the target has been
    re-evaluated, so `sweepeval gate --profile quick` paid for a full
    evaluation -- 120 requests against a live endpoint, measured -- and then
    refused on a property of the command line that was knowable before the
    first request. §3 makes cost a first-class constraint and I9 puts the
    estimate before any spend; charging for a refusal inverts both.

    ``quick``'s cluster counts sit just above the bootstrap floor, so its
    intervals are too wide to gate on. Refusing is better than passing
    everything and calling it a green build.
    """
    if profile == "quick":
        return (
            "profile=quick is not gate-eligible: its intervals are wide by "
            "design. Re-run with --profile standard."
        )
    return None


DEFAULT_GATE_ON = (
    "security_pass_rate",
    "guardrail_pass_rate",
    "target_determinism_at_temp0",
    "context_retention_auc",
    "cost_per_probe",
)
"""§16's default: the six objectives minus latency.

Latency is deliberately excluded. Between-session network and server variance
is 20-50%, far above what the measurement can attribute to the target, so
gating it flaps for reasons that have nothing to do with the code under test.
``--gate-on latency_p95_ms`` opts in.

Determinism, retention and cost were missing from this tuple, so a regression
in any of the three never failed a default gate -- and the one test covering
it asserted only that latency was absent, which stayed green while four other
required metrics were too.
"""


def snapshot(
    result: EvaluationResult, *, hard_fails: tuple[str, ...] | None = None
) -> Baseline:
    """Turn a result into a committable baseline.

    ``hard_fails`` defaults to the ones the result itself recorded. It used to
    default to ``()`` and no caller ever passed anything, so §16's "any
    security hard-fail fails the gate regardless of the baseline" was
    unreachable.
    """
    return Baseline(
        run_id=result.run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        config_id=result.config_id,
        comparability=result.comparability,
        metrics=dict(result.metrics),
        clusters={k: dict(v) for k, v in result.clusters.items()},
        strata={k: dict(v) for k, v in (getattr(result, 'strata', None) or {}).items()},
        hard_fails=(
            hard_fails
            if hard_fails is not None
            else getattr(result, "hard_fails", None) or ()
        ),
    )


def save_baseline(baseline: Baseline, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = baseline.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def load_baseline(path: Path) -> Baseline:
    return Baseline.model_validate_json(Path(path).read_text(encoding="utf-8"))


def gate(
    result: EvaluationResult,
    baseline: Baseline | Path | str,
    *,
    alpha: float = DEFAULT_GATE_ALPHA,
    gate_on: tuple[str, ...] | None = None,
    min_effect_overrides: dict[str, float] | None = None,
    seed: int = 0,
    objectives: tuple[Objective, ...] | None = None,
) -> GateVerdict:
    """Compare a result against a baseline. Pure; sends nothing."""
    if not isinstance(baseline, Baseline):
        baseline = load_baseline(Path(baseline))

    verdict = _check_comparability(result, baseline)
    if verdict is not None:
        return verdict

    refusal = profile_refusal(result.profile)
    if refusal is not None:
        return GateVerdict(
            ok=False, exit_code=ExitCode.USAGE_ERROR, refusals=(refusal,)
        )

    objectives = objectives or REGISTRY.defaults()
    selected = gate_on if gate_on is not None else DEFAULT_GATE_ON

    return gate_metrics(
        objectives,
        baseline.clusters,
        result.clusters,
        # The CURRENT run's leaks, not the baseline's. Passing the baseline's
        # inverted the check: a run that started leaking passed, and one whose
        # baseline had leaked failed forever after it was fixed.
        hard_fails=tuple(getattr(result, "hard_fails", None) or ()),
        alpha=alpha,
        seed=seed,
        min_effect_overrides=min_effect_overrides,
        gate_on=selected,
        # The baseline's labels, not the current run's: the paired test runs
        # over clusters both sides scored, and the weights have to be the ones
        # the committed baseline was computed with or the two points are not
        # on the same curve.
        strata=baseline.strata or {
            k: dict(v) for k, v in (getattr(result, "strata", None) or {}).items()
        },
    )


def _check_comparability(
    result: EvaluationResult, baseline: Baseline
) -> GateVerdict | None:
    """I6, before any statistics run."""
    decision = compare_keys(baseline.comparability, result.comparability)
    if decision.ok:
        return None
    return GateVerdict(
        ok=False,
        exit_code=ExitCode.COMPARABILITY_REFUSED,
        refusals=tuple(m.message for m in decision.refusals),
        notes=(
            "this is not a regression — the two runs measured different "
            "things. Re-baseline, or restore the setting that changed.",
        ),
    )


def gate_payload(verdict: GateVerdict) -> dict[str, Any]:
    """Machine-readable gate result, for --format json and the Action."""
    return {
        "ok": verdict.ok,
        "exit_code": verdict.exit_code.value,
        "regressions": [
            {
                "metric": d.metric,
                "baseline": d.baseline_point,
                "current": d.current_point,
                "difference": d.difference,
                "min_effect": d.min_effect,
                "p_value": d.p_value,
                "threshold": d.adjusted_threshold,
            }
            for d in verdict.regressions
        ],
        "metrics": [
            {
                "metric": d.metric,
                "baseline": d.baseline_point,
                "current": d.current_point,
                "difference": d.difference,
                "regressed": d.regressed,
                "reason": d.reason,
            }
            for d in verdict.diffs
        ],
        "hard_fails": list(verdict.hard_fails),
        "refusals": list(verdict.refusals),
        # What the gate could NOT do. Absent from the payload entirely
        # before, so a CI job parsing gate.json had no way to tell a run
        # that tested five metrics from one that tested one.
        "not_gated": list(verdict.not_gated),
        "degraded": list(verdict.degraded),
        "incomplete": verdict.incomplete,
    }


__all__ += ["gate_payload"]
