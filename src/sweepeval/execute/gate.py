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
    "model_to_pin",
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
        model=getattr(result, "model", None),
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
    allow_model_change: bool = False,
) -> GateVerdict:
    """Compare a result against a baseline. Pure; sends nothing."""
    if not isinstance(baseline, Baseline):
        baseline = load_baseline(Path(baseline))

    verdict = _gate(
        result, baseline,
        alpha=alpha, gate_on=gate_on, min_effect_overrides=min_effect_overrides,
        seed=seed, objectives=objectives, allow_model_change=allow_model_change,
    )
    # Stamped on every exit, including the refusals: a CI job reading gate.json
    # needs to know which models the verdict is about whatever the verdict was,
    # and four separate `return`s is how one of them ends up without it.
    verdict.baseline_model = baseline.model
    verdict.current_model = getattr(result, "model", None)
    return verdict


def _gate(
    result: EvaluationResult,
    baseline: Baseline,
    *,
    alpha: float,
    gate_on: tuple[str, ...] | None,
    min_effect_overrides: dict[str, float] | None,
    seed: int,
    objectives: tuple[Objective, ...] | None,
    allow_model_change: bool,
) -> GateVerdict:
    verdict = _check_comparability(result, baseline)
    if verdict is not None:
        return verdict

    model_verdict, model_notes = _check_model(result, baseline, allow_model_change)
    if model_verdict is not None:
        return model_verdict

    refusal = profile_refusal(result.profile)
    if refusal is not None:
        return GateVerdict(
            ok=False, exit_code=ExitCode.USAGE_ERROR, refusals=(refusal,)
        )

    objectives = objectives or REGISTRY.defaults()
    selected = gate_on if gate_on is not None else DEFAULT_GATE_ON

    verdict = gate_metrics(
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
    # Carried onto a verdict the statistics produced, not dropped: a gate that
    # could not check the model, or was told to ignore a change, has to say so
    # on the same output that reports the numbers.
    verdict.notes = verdict.notes + model_notes
    return verdict


def model_to_pin(
    baseline: Baseline, requested: str | None, allow_model_change: bool = False
) -> str | None:
    """Which model a gate run should pin, given the baseline and the flags.

    Defaulting to the baseline's model is what makes the gate re-measure what
    the baseline measured. Without it the gate spends a full run -- 120
    requests at `standard` -- and only then discovers its model is not the
    baseline's and refuses; §3 makes cost a first-class constraint.

    Shared by the CLI and :func:`sweepeval.api.arun_gate`, which are the two
    ways to re-run and gate. They had no shared helper and the second would
    have quietly kept the old behaviour.
    """
    if requested is not None:
        return requested
    if allow_model_change:
        return None
    return baseline.model or None


def _check_model(
    result: EvaluationResult, baseline: Baseline, allow_model_change: bool
) -> tuple[GateVerdict | None, tuple[str, ...]]:
    """Refuse a gate whose model is not the baseline's (§16).

    Not part of :func:`_check_comparability`, because the model is not a
    comparability key and must not become one -- a sweep varies it across
    configs inside one run. It is a property of the single config a baseline
    and a gate each measure, and comparing two different models is the same
    category of mistake as comparing two different profiles: the number moves
    for a reason the change under test did not cause.

    Three cases that are not a refusal:

    * neither side named a model -- a shape that puts it in the URL, or has no
      notion of one. Nothing to compare, and nothing is claimed.
    * the baseline predates the field. It cannot be checked, and silently
      treating "unknown" as "matching" is what this whole check exists to stop,
      so it is annotated instead.
    * ``allow_model_change``, which is how you deliberately gate a model
      upgrade. Still annotated: a green gate that compared two models must not
      look like a green gate that compared one.
    """
    was, now = baseline.model, getattr(result, "model", None)
    if was == now:
        return None, ()
    if was is None:
        return None, (
            f"this baseline predates model recording, so the gate could not "
            f"check that it measured the same model as this run "
            f"({now!r}). Re-baseline to make the check active.",
        )
    if now is None:
        return None, (
            f"the baseline named model {was!r} and this run's requests name no "
            "model at all, so the gate could not check them against each other.",
        )
    if allow_model_change:
        return None, (
            f"--allow-model-change: gated {was!r} against {now!r}. Any "
            "difference below may be the model, not the change under test.",
        )
    return (
        GateVerdict(
            ok=False,
            exit_code=ExitCode.COMPARABILITY_REFUSED,
            refusals=(
                f"model differs: {was} vs {now} — the baseline and this run "
                "measured different models, so a difference between them is "
                "not evidence about the change under test",
            ),
            notes=(
                "re-baseline against this model, pin the baseline's model with "
                "--model, or pass --allow-model-change to compare them anyway.",
            ),
        ),
        (),
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
        "model": {
            "baseline": verdict.baseline_model,
            "current": verdict.current_model,
        },
        # What the gate could NOT do. Absent from the payload entirely
        # before, so a CI job parsing gate.json had no way to tell a run
        # that tested five metrics from one that tested one.
        "not_gated": list(verdict.not_gated),
        "degraded": list(verdict.degraded),
        "incomplete": verdict.incomplete,
    }


__all__ += ["gate_payload"]
