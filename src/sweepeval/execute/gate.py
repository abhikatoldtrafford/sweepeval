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

__all__ = ["DEFAULT_GATE_ON", "gate", "load_baseline", "save_baseline", "snapshot"]

DEFAULT_GATE_ON = (
    "security_pass_rate",
    "guardrail_pass_rate",
)
"""§16's default. Latency is deliberately excluded.

Between-session network and server variance is 20-50%, far above what the
measurement can attribute to the target, so gating it flaps for reasons that
have nothing to do with the code under test. ``--gate-on latency_p95_ms`` opts
in.
"""


def snapshot(result: EvaluationResult, *, hard_fails: tuple[str, ...] = ()) -> Baseline:
    """Turn a result into a committable baseline."""
    return Baseline(
        run_id=result.run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        config_id=result.config_id,
        comparability=result.comparability,
        metrics=dict(result.metrics),
        clusters={k: dict(v) for k, v in result.clusters.items()},
        hard_fails=hard_fails,
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

    if result.profile == "quick":
        # §16: quick's cluster counts sit just above the bootstrap floor, so
        # its intervals are too wide to gate on. Refusing is better than
        # passing everything and calling it a green build.
        return GateVerdict(
            ok=False,
            exit_code=ExitCode.USAGE_ERROR,
            refusals=(
                "profile=quick is not gate-eligible: its intervals are wide by "
                "design. Re-run with --profile standard.",
            ),
        )

    objectives = objectives or REGISTRY.defaults()
    selected = gate_on if gate_on is not None else DEFAULT_GATE_ON

    return gate_metrics(
        objectives,
        baseline.clusters,
        result.clusters,
        hard_fails=baseline.hard_fails,
        alpha=alpha,
        seed=seed,
        min_effect_overrides=min_effect_overrides,
        gate_on=selected,
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
    }


__all__ += ["gate_payload"]
