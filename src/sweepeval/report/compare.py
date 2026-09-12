"""``sweepeval compare`` (spec §4.1, §6.5). I6 load-bearing.

I6: "Results whose hard comparability keys differ refuse to be compared."

A pure function over two stored manifests. It sends nothing, re-runs nothing,
and needs no credentials — which is what makes it safe to point at a colleague's
committed run.

The refusal is never the string "results incomparable". It names the key and
both values, because a user told two runs cannot be compared needs to know
which of eleven things changed, and usually it is one they can undo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sweepeval.schema.comparability import Comparability, KeyMismatch, compare_keys

__all__ = ["ComparisonResult", "compare_manifests", "compare_runs", "load_manifest"]


@dataclass
class ComparisonResult:
    ok: bool
    refusals: tuple[KeyMismatch, ...] = ()
    warnings: tuple[KeyMismatch, ...] = ()
    a: str = ""
    b: str = ""
    deltas: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    """Metric -> (a's point, b's point). Only populated when the keys match."""

    def explain(self) -> str:
        if self.ok and not self.warnings:
            return f"{self.a} and {self.b} are comparable"
        lines: list[str] = []
        if not self.ok:
            lines.append(
                f"refusing to compare {self.a} with {self.b} — these runs do not "
                "measure the same thing:"
            )
            lines.extend(f"  {m.message}" for m in self.refusals)
        for mismatch in self.warnings:
            lines.append(f"  warning: {mismatch.message}")
        return "\n".join(lines)


def load_manifest(path: Path | str) -> dict[str, Any]:
    """Read a run's manifest, whether given the run directory or the file."""
    from sweepeval.store.json_io import read_json

    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "manifest.json"
    payload = read_json(candidate)
    if not payload:
        raise FileNotFoundError(f"no readable manifest at {candidate}")
    return payload


def compare_manifests(a: dict[str, Any], b: dict[str, Any]) -> ComparisonResult:
    """Decide whether two runs may be compared (I6)."""
    keys_a = a.get("comparability")
    keys_b = b.get("comparability")
    run_a = str(a.get("run_id", "(unknown)"))
    run_b = str(b.get("run_id", "(unknown)"))

    missing = [
        name
        for name, keys in ((run_a, keys_a), (run_b, keys_b))
        if not isinstance(keys, dict)
    ]
    if missing:
        return ComparisonResult(
            ok=False,
            a=run_a,
            b=run_b,
            refusals=(
                KeyMismatch(
                    key="comparability",
                    a="(absent)" if not isinstance(keys_a, dict) else "present",
                    b="(absent)" if not isinstance(keys_b, dict) else "present",
                    message=(
                        f"{', '.join(missing)} has no comparability keys, so there "
                        "is nothing to check this comparison against"
                    ),
                ),
            ),
        )

    verdict = compare_keys(
        Comparability.model_validate(keys_a), Comparability.model_validate(keys_b)
    )
    return ComparisonResult(
        ok=verdict.ok,
        refusals=verdict.refusals,
        warnings=verdict.warnings,
        a=run_a,
        b=run_b,
    )


def compare_runs(a: Path | str, b: Path | str) -> ComparisonResult:
    """Compare two run directories by their manifests."""
    return compare_manifests(load_manifest(a), load_manifest(b))
