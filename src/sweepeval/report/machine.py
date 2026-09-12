"""Machine-readable reports: JSON, Markdown, GitHub annotations (spec §15, §30).

Written only when asked (``--format``), never by default. GitHub Actions
annotations auto-enable under ``GITHUB_ACTIONS``, because that is the one case
where the user cannot see the terminal.

Every renderer shows the same three things the terminal does and for the same
reason: intervals, skips with reasons, and assumptions. A Markdown summary that
omits the SKIPPED block is a report that looks complete and is not.
"""

from __future__ import annotations

import json
import os
from typing import Any

from sweepeval.schema.metric import CIMethod, MetricValue

__all__ = ["as_github_annotations", "as_json", "as_markdown", "in_github_actions"]


def in_github_actions() -> bool:
    return os.environ.get("GITHUB_ACTIONS", "").lower() == "true"


def _metric_payload(value: MetricValue) -> dict[str, Any]:
    from sweepeval.schema.metric import Flag

    measured = Flag.NO_VALID_INTERVAL not in value.flags
    return {
        "point": value.point if measured else None,
        "measured": measured,
        "lo": value.lo,
        "hi": value.hi,
        "method": value.method.value,
        "n_clusters": value.n_clusters,
        "alpha": value.alpha,
        "estimand": value.estimand.value,
        "flags": [f.value for f in value.flags],
    }


def as_json(result: Any) -> str:
    payload = {
        "run_id": result.run_id,
        "config_id": result.config_id,
        "profile": result.profile,
        "single_config": True,
        "target": result.discovery.payload["target"],
        "extraction": result.discovery.payload["extraction"],
        "corpus": {
            "units": result.corpus.unit_count,
            "calls_per_run": result.corpus.calls_per_run,
            "hash": result.corpus.hash,
        },
        "comparability": result.comparability.model_dump(mode="json"),
        "capabilities": result.capabilities.to_manifest(),
        "metrics": {k: _metric_payload(v) for k, v in sorted(result.metrics.items())},
        "skipped": [{"family": f, "reason": r} for f, r in sorted(result.skipped)],
        "assumptions": [
            {"field": f, "confidence": c, "detail": d} for f, c, d in result.assumptions
        ],
        "axes_rejected": [{"axis": a, "reason": r} for a, r in result.axes_rejected],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _interval(value: MetricValue) -> str:
    if value.method is CIMethod.none or value.lo is None or value.hi is None:
        return "no valid interval"
    return f"[{value.lo:.3g}, {value.hi:.3g}]"


def _point(value: MetricValue) -> str:
    """See terminal._point: an unmeasurable metric shows no number."""
    from sweepeval.schema.metric import Flag

    if Flag.NO_VALID_INTERVAL in value.flags:
        return "—"
    return f"{value.point:.4g}"


def as_markdown(result: Any) -> str:
    lines: list[str] = [
        "# sweepeval — single-configuration evaluation",
        "",
        f"- **target** `{result.discovery.payload['target']['url']}`",
        f"- **shape** `{result.discovery.ladder.shape.name}`",
        f"- **type** {result.discovery.target_type}",
        f"- **profile** {result.profile} "
        f"({result.corpus.unit_count} units, {result.corpus.calls_per_run} calls/run)",
        f"- **run** `{result.run_id}`",
        "",
        "## Metrics",
        "",
        "| metric | value | 95% interval | clusters | flags |",
        "|---|---:|---:|---:|---|",
    ]
    for name in sorted(result.metrics):
        value = result.metrics[name]
        flags = ", ".join(f.value for f in value.flags) or "—"
        lines.append(
            f"| `{name}` | {_point(value)} | {_interval(value)} | "
            f"{value.n_clusters} | {flags} |"
        )

    if result.axes_rejected:
        lines += [
            "",
            "## No variable axes discovered",
            "",
            "This is a single configuration, not a sweep. Candidate axes and why "
            "each was rejected:",
            "",
        ]
        lines += [f"- `{axis}` — {reason}" for axis, reason in result.axes_rejected]

    if result.skipped:
        lines += ["", "## Skipped", ""]
        lines += [f"- `{family}` — {reason}" for family, reason in sorted(result.skipped)]

    if result.assumptions:
        lines += ["", "## Assumptions you can correct", ""]
        lines += [
            f"- `{field}` ({confidence}) — {detail}"
            for field, confidence, detail in result.assumptions
        ]

    return "\n".join(lines) + "\n"


def as_github_annotations(result: Any) -> str:
    """Workflow-command lines. Warnings only — an evaluation is not a failure."""
    lines: list[str] = []
    for field, confidence, detail in result.assumptions:
        lines.append(
            f"::warning title=sweepeval assumption::{field} ({confidence}) — {detail}"
        )
    for family, reason in sorted(result.skipped):
        lines.append(f"::notice title=sweepeval skipped::{family} — {reason}")
    return "\n".join(lines) + ("\n" if lines else "")


def gate_annotations(payload: dict[str, Any]) -> str:
    """Annotations for a gate result. Regressions are errors."""
    lines: list[str] = []
    for regression in payload.get("regressions", []):
        lines.append(
            "::error title=sweepeval regression::"
            f"{regression['metric']} {regression['baseline']:.4g} -> "
            f"{regression['current']:.4g} "
            f"(p={regression['p_value']:.3g})"
        )
    for hard in payload.get("hard_fails", []):
        lines.append(f"::error title=sweepeval hard fail::{hard}")
    for refusal in payload.get("refusals", []):
        lines.append(f"::error title=sweepeval refused::{refusal}")
    return "\n".join(lines) + ("\n" if lines else "")


__all__ += ["gate_annotations"]
