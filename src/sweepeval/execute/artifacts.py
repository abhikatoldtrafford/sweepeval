"""``plan.json``, ``manifest.json`` and the resume check (spec §6.2, §12.5).

The plan is written **before** the first scoring request and never rewritten.
That is what makes I4 checkable from the artifact rather than from the code:
open ``plan.json``, and the serialised Units and the canary table are right
there, identical for every config in the sweep. A plan reconstructed at report
time from whatever the run happened to do would prove nothing.

``--resume`` verifies the plan hash, the corpus hash and **every** hard
comparability key before appending a single row. The failure it exists to
prevent is quiet: a user edits their config, resumes, and the finished run
mixes rows measured under two different definitions with nothing in the
artifact saying so.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sweepeval.execute.planner import SweepPlan
from sweepeval.schema.comparability import Comparability, KeyMismatch, compare_keys
from sweepeval.schema.hashing import hash_obj
from sweepeval.schema.unit import Unit
from sweepeval.schema.versions import SCHEMA_MAJOR, SUITE_VERSION, TOOL_VERSION

__all__ = [
    "PlanDocument",
    "ResumeVerdict",
    "build_manifest",
    "build_plan_document",
    "read_json",
    "verify_resume",
    "write_json",
]


@dataclass
class PlanDocument:
    """The frozen plan: configs, axes, units and canaries (§12.1, I4)."""

    run_id: str
    profile: str
    runs: int
    master_seed: str
    corpus_hash: str
    configs: list[dict[str, Any]] = field(default_factory=list)
    axes: dict[str, list[Any]] = field(default_factory=dict)
    shrink_steps: list[str] = field(default_factory=list)
    rejected_axes: list[list[str]] = field(default_factory=list)
    dropped_models: list[list[str]] = field(default_factory=list)
    cap: int = 12
    units: list[dict[str, Any]] = field(default_factory=list)
    canary_table: dict[str, str] = field(default_factory=dict)
    determinism_sharing: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "profile": self.profile,
            "runs": self.runs,
            "master_seed": self.master_seed,
            "corpus_hash": self.corpus_hash,
            "cap": self.cap,
            "axes": {k: list(v) for k, v in sorted(self.axes.items())},
            "shrink_steps": list(self.shrink_steps),
            "rejected_axes": [list(r) for r in self.rejected_axes],
            "dropped_models": [list(d) for d in self.dropped_models],
            "configs": self.configs,
            "units": self.units,
            "canary_table": dict(sorted(self.canary_table.items())),
            "determinism_sharing": dict(sorted(self.determinism_sharing.items())),
        }

    def hash(self) -> str:
        """Identity of the experiment, not of the run that executed it.

        ``run_id`` and the canary values derived from it are excluded: a
        resumed run reuses both, but including them would make the hash
        useless for the one comparison it is for — this plan against the plan
        the user's edited config would produce now.
        """
        payload = self.to_dict()
        payload.pop("run_id")
        payload.pop("master_seed")
        payload.pop("canary_table")
        return hash_obj(payload)


def build_plan_document(
    plan: SweepPlan,
    units: Sequence[Unit],
    canaries: Mapping[tuple[str, int, str], str],
    *,
    run_id: str,
    profile: str,
    runs: int,
    master_seed: str,
    corpus_hash: str,
    determinism_sharing: Mapping[str, str] | None = None,
) -> PlanDocument:
    """Serialise the frozen plan.

    The Units and the canary table are written **once**, outside the config
    list, precisely because they are shared. Writing them per config would
    make an I4 violation representable in the artifact: two configs could
    carry different probe sets and the file would look fine.
    """
    return PlanDocument(
        run_id=run_id,
        profile=profile,
        runs=runs,
        master_seed=master_seed,
        corpus_hash=corpus_hash,
        cap=plan.cap,
        axes={k: list(v) for k, v in plan.axes.items()},
        shrink_steps=list(plan.shrink_steps),
        rejected_axes=[[a, r] for a, r in plan.rejected_axes],
        dropped_models=[[m, r] for m, r in plan.dropped_models],
        configs=[
            {
                "config_id": c.config_id,
                "label": c.label(),
                "params": dict(sorted(c.params.items())),
                "system_prompt_variant": c.system_prompt_variant,
                "system_prompt": c.system_prompt,
            }
            for c in plan.configs
        ],
        units=[u.model_dump(mode="json") for u in units],
        canary_table={
            f"{unit_id}|{run_idx}|{name}": value
            for (unit_id, run_idx, name), value in sorted(canaries.items())
        },
        determinism_sharing=dict(determinism_sharing or {}),
    )


def build_manifest(
    *,
    run_id: str,
    comparability: Comparability,
    plan_hash: str,
    corpus_hash: str,
    capabilities: Mapping[str, Any],
    target: Mapping[str, Any],
    authorization: Mapping[str, str] | None,
    seed: int,
    master_seed: str,
    budget: Mapping[str, Any],
    heuristics: Mapping[str, str],
) -> dict[str, Any]:
    """The manifest of §6.2: what was run, against what, and under which rules."""
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tool_version": TOOL_VERSION,
        "schema_major": SCHEMA_MAJOR,
        "suite_version": SUITE_VERSION,
        "comparability": comparability.model_dump(mode="json"),
        "plan_hash": plan_hash,
        "corpus_hash": corpus_hash,
        "capabilities": dict(capabilities),
        "target": dict(target),
        "authorization": dict(authorization) if authorization else None,
        "seeds": {"seed": seed, "master_seed": master_seed},
        "budget": dict(budget),
        "heuristics": dict(heuristics),
    }


@dataclass
class ResumeVerdict:
    """Whether an existing run may be continued (§12.5)."""

    ok: bool
    refusals: tuple[KeyMismatch, ...] = ()
    warnings: tuple[KeyMismatch, ...] = ()
    requests_already_spent: int = 0
    tokens_already_spent: int = 0

    def explain(self) -> str:
        if self.ok and not self.warnings:
            return "resume: the plan, the corpus and every hard key match"
        lines = []
        for mismatch in self.refusals:
            lines.append(f"refused: {mismatch.message}")
        for mismatch in self.warnings:
            lines.append(f"warning: {mismatch.message}")
        if self.ok:
            lines.insert(0, "resume: proceeding despite soft-key differences")
        else:
            lines.insert(
                0,
                "refusing to resume: this run would mix rows measured under two "
                "different definitions",
            )
        return "\n".join(lines)


def verify_resume(
    existing: Mapping[str, Any],
    *,
    comparability: Comparability,
    plan_hash: str,
    corpus_hash: str,
) -> ResumeVerdict:
    """Compare a stored manifest against what this invocation would produce."""
    refusals: list[KeyMismatch] = []

    stored_plan = str(existing.get("plan_hash", ""))
    if stored_plan != plan_hash:
        refusals.append(
            KeyMismatch(
                key="plan_hash",
                a=stored_plan or "(absent)",
                b=plan_hash,
                message=(
                    f"plan hash differs: {_short(stored_plan)} vs {_short(plan_hash)} "
                    "— the sweep's configs, axes or probe set changed since this run "
                    "started, so its completed rows and its remaining ones would not "
                    "be the same experiment"
                ),
            )
        )

    stored_corpus = str(existing.get("corpus_hash", ""))
    if stored_corpus != corpus_hash:
        refusals.append(
            KeyMismatch(
                key="corpus_hash",
                a=stored_corpus or "(absent)",
                b=corpus_hash,
                message=(
                    f"corpus hash differs: {_short(stored_corpus)} vs "
                    f"{_short(corpus_hash)} — the probe corpus changed since this "
                    "run started"
                ),
            )
        )

    warnings: tuple[KeyMismatch, ...] = ()
    stored_keys = existing.get("comparability")
    if isinstance(stored_keys, Mapping):
        stored = Comparability.model_validate(dict(stored_keys))
        verdict = compare_keys(stored, comparability)
        refusals.extend(verdict.refusals)
        warnings = verdict.warnings
    else:
        refusals.append(
            KeyMismatch(
                key="comparability",
                a="(absent)",
                b="present",
                message=(
                    "the stored manifest has no comparability keys, so there is "
                    "nothing to verify this resume against"
                ),
            )
        )

    return ResumeVerdict(
        ok=not refusals, refusals=tuple(refusals), warnings=tuple(warnings)
    )


def _short(value: str) -> str:
    return f"{value[:12]}…" if len(value) > 14 else (value or "(absent)")


def write_json(path: Path, payload: Any) -> None:
    """Write a metadata artifact readably and deterministically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}
