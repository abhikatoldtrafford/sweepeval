"""Re-score a stored run offline (spec §5.1, I7).

§5.1 sells the store as re-scorable without paying again, and until now
nothing exposed that. When three security verdicts in the published 14-model
run turned out to be wrong, correcting them meant a one-off script — and the
numbers on `docs/scorecard.md` came from something no reader could run.

Sends nothing and needs no credentials. Every scored observation records the
blob address of the text it scored, so the verdict can be recomputed from the
kept response: read the blob, re-derive the canary from the manifest's master
seed, and call the scorer this build ships.

**It never rewrites `observations.jsonl`.** That log is what was paid for and
it is append-only (I7); a re-score is a derived view of it. What comes back is
a report of what *would* change, plus the re-aggregated metrics, and the
caller decides what to do with them.

The check that makes it trustworthy is in :meth:`RescoreReport.reproduced`:
re-aggregating the configs whose verdicts did **not** change must land on the
numbers the run itself stored. If it does not, the offline path differs from
the one that produced the run, and the corrected figures would be a different
measurement rather than a fix.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sweepeval.corpus.loader import load_corpus
from sweepeval.execute.aggregation import aggregate_config
from sweepeval.schema.metric import MetricValue
from sweepeval.schema.observation import Observation, Verdict
from sweepeval.schema.unit import Unit
from sweepeval.scorers.base import ScoreContext, ScorerRegistry
from sweepeval.store.redaction import Redactor
from sweepeval.store.run import Store

__all__ = ["Change", "RescoreReport", "rescore_run"]

RESCORABLE = ("security", "guardrail")
"""Families a stored run can be re-scored for.

Both decide from the final response text alone, which is the thing the store
keeps. `determinism` needs every run of a unit at once; its rows only started
carrying blob addresses recently, so older runs cannot be re-scored for it at
all and newer ones are not attempted here yet. `operational` derives from
`calls.jsonl` rather than from text and never needs this.
"""


@dataclass(frozen=True)
class Change:
    """One verdict that the current scorer disagrees with."""

    config_id: str
    unit_id: str
    run_idx: int
    metric: str
    was: str
    now: str
    reason: str

    def __str__(self) -> str:
        return (
            f"{self.config_id} {self.unit_id.split('#')[0]} run{self.run_idx} "
            f"{self.metric}: {self.was} -> {self.now} ({self.reason})"
        )


@dataclass
class RescoreReport:
    run_dir: Path
    families: tuple[str, ...]
    changes: list[Change] = field(default_factory=list)
    metrics: dict[str, dict[str, MetricValue]] = field(default_factory=dict)
    considered: int = 0
    unjoinable: int = 0
    """Scored rows whose text is not in the store, so nothing can be recomputed.

    Never silently skipped: a re-score that quietly ignored them would report
    "nothing changed" for a run it could not read.
    """

    reproduced: int = 0
    mismatched: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatched

    @property
    def touched(self) -> set[str]:
        return {c.config_id for c in self.changes}


def rescore_run(
    run_dir: Path | str,
    *,
    families: Sequence[str] = RESCORABLE,
    seed: int = 0,
    registry: ScorerRegistry | None = None,
) -> RescoreReport:
    """Recompute stored verdicts with this build's scorers. Sends nothing."""
    from sweepeval.scorers import registry as default_registry

    directory = Path(run_dir)
    unknown = sorted(set(families) - set(RESCORABLE))
    if unknown:
        raise ValueError(
            f"cannot re-score {', '.join(unknown)} from stored text; "
            f"re-scorable families are {', '.join(RESCORABLE)}"
        )

    plan = _plan_of(directory)
    canary_table = _canary_table(plan)
    profile = plan.get("profile") or "standard"
    # No secrets to mask: this reads and never writes, and the log it
    # reads was redacted when it was written.
    store = Store(directory.parent.parent, directory.name, Redactor())
    scorers = registry or default_registry()

    units = {}
    for template in load_corpus(profile=profile).probes:
        unit = template.to_unit()
        units[unit.unit_id] = unit

    report = RescoreReport(run_dir=directory, families=tuple(families))
    rows: dict[str, list[Observation]] = {}

    for line in (directory / "observations.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        replacement = _rescore_row(
            row, units, store, scorers, canary_table, families, report
        )
        rows.setdefault(row["config_id"], []).append(replacement)

    for config_id, observations in rows.items():
        report.metrics[config_id] = dict(
            aggregate_config(observations, config_id=config_id, seed=seed).metrics
        )

    _check_reproduction(directory, report, rows, seed)
    return report


def _rescore_row(
    row: dict[str, Any],
    units: dict[str, Unit],
    store: Store,
    scorers: ScorerRegistry,
    canary_table: dict[str, str],
    families: Sequence[str],
    report: RescoreReport,
) -> Observation:
    if row.get("family") not in families or row["verdict"] not in ("PASS", "FAIL"):
        return Observation.model_validate(row)

    unit = units.get(row["unit_id"])
    text = _text_of(row, store)
    if unit is None or text is None:
        report.unjoinable += 1
        return Observation.model_validate(row)

    report.considered += 1
    canaries = {
        name: canary_table[key]
        for name in unit.canary_names
        if (key := f"{row['unit_id']}|{row['run_idx']}|{name}") in canary_table
    }
    context = ScoreContext(
        run_id=row["run_id"], config_id=row["config_id"],
        run_idx=row["run_idx"], text=text, canaries=canaries,
        ts=row["ts"], layer=row.get("layer") or "generic",
    )
    fresh = [
        o
        for o in scorers.get(row["family"]).score(unit, [], context)
        if o.metric == row["metric"]
    ]
    if not fresh:
        return Observation.model_validate(row)

    now = fresh[0]
    if now.verdict.value != row["verdict"]:
        report.changes.append(
            Change(
                config_id=row["config_id"], unit_id=row["unit_id"],
                run_idx=row["run_idx"], metric=row["metric"],
                was=row["verdict"], now=now.verdict.value,
                reason=now.reason or "",
            )
        )
    return now


def _text_of(row: dict[str, Any], store: Store) -> str | None:
    for blob_id in row.get("blob_ids") or ():
        try:
            return store.blobs.get(blob_id).decode("utf-8")
        except (KeyError, UnicodeDecodeError):
            continue
    return None


def _plan_of(directory: Path) -> dict[str, Any]:
    """The run's plan, which is what makes a re-score possible at all.

    `plan.json` holds the canary table the run actually used, keyed
    ``unit_id|run_idx|name``. Deriving the values again from a master seed
    would work too, but reading the recorded ones cannot drift from what was
    sent.

    A run without it cannot be re-scored: canary values are per-run, and a
    security verdict recomputed against the wrong canary is worse than no
    verdict. `sweep` writes one; a bare `aevaluate_target` does not.
    """
    path = directory / "plan.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no plan.json in {directory} — a re-score needs the canary table "
            "the run used, and only `sweepeval sweep` writes one. There is "
            "nothing to recover it from."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("payload") or payload


def _canary_table(plan: dict[str, Any]) -> dict[str, str]:
    table = plan.get("canary_table")
    if not isinstance(table, dict) or not table:
        raise FileNotFoundError(
            "plan.json carries no canary table, so security verdicts cannot "
            "be recomputed against the values the run actually sent"
        )
    return {str(k): str(v) for k, v in table.items()}


def _check_reproduction(
    directory: Path,
    report: RescoreReport,
    rows: dict[str, list[Observation]],
    seed: int,
) -> None:
    """Re-aggregating an untouched config must reproduce what the run stored.

    Without this the corrected numbers are only as trustworthy as the claim
    that the offline path matches the one that produced them, and that claim
    would be untested precisely where it matters.
    """
    aggregates = directory / "aggregates.json"
    if not aggregates.is_file():
        return
    stored = json.loads(aggregates.read_text(encoding="utf-8"))
    payload = stored.get("payload") or stored

    for config in payload.get("configs", []):
        config_id = config["config_id"]
        if config_id in report.touched or config_id not in rows:
            continue
        fresh = report.metrics[config_id]
        for metric, was in (config.get("metrics") or {}).items():
            now = fresh.get(metric)
            if was.get("point") is None or now is None or now.point is None:
                continue
            if abs(was["point"] - now.point) > 1e-9:
                report.mismatched.append(
                    f"{config_id}.{metric}: stored {was['point']!r} "
                    f"vs offline {now.point!r}"
                )
            else:
                report.reproduced += 1


def verdict_counts(observations: Sequence[Observation], metric: str) -> dict[str, int]:
    """PASS/FAIL/UNSCORABLE counts for one metric, for the report's table."""
    counts: dict[str, int] = {v.value: 0 for v in Verdict}
    for observation in observations:
        if observation.metric == metric:
            counts[observation.verdict.value] += 1
    return counts
