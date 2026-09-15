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
from sweepeval.judge.escalate import JUDGE_SCORER
from sweepeval.schema.metric import MetricValue
from sweepeval.schema.observation import Observation, Verdict
from sweepeval.schema.unit import Unit
from sweepeval.scorers.base import ScoreContext, ScorerRegistry
from sweepeval.store.redaction import Redactor
from sweepeval.store.run import Store

__all__ = [
    "Change",
    "RescoreReport",
    "is_escalatable",
    "rescore_run",
    "text_of",
    "units_for",
]

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

    rows: dict[str, list[Observation]] = field(default_factory=dict)
    """The re-scored observations, per config, as they would be aggregated.

    Returned rather than discarded so a caller can do something with them
    without re-reading and re-scoring the run -- `rejudge` escalates the
    ambiguous ones. Still nothing is written; this is the derived view in
    memory.
    """

    escalatable: int = 0
    """Ambiguities an LLM judge could be asked to resolve (§11.9).

    Counted on every re-score, including one with no judge configured, because
    the number is free to compute and is the one a reader needs in order to
    decide whether a judge is worth its calls. Only rows whose response text
    is still in the store count: an ambiguity with nothing to show the judge
    is not resolvable at any price.
    """

    unjudgeable: int = 0
    """Ambiguities with no stored response, so no judge can settle them.

    Reported rather than subtracted in silence: a tool that says "420 to
    judge" when there are 421 ambiguities has quietly dropped the case this
    whole feature exists to make visible.

    It reads zero on the published 14-model run. It did not always: every
    re-scored row lost its ``blob_ids``, because a scorer is handed text
    rather than addresses, so one ambiguity that the current scorer produces
    looked like an ambiguity with no evidence behind it. The count is only
    meaningful because the re-score carries provenance forward.
    """

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

    units = units_for(profile)

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

    report.rows = rows
    for observations in rows.values():
        for o in observations:
            if not _is_ambiguous(o):
                continue
            if is_escalatable(o) and text_of(o, store) is not None:
                report.escalatable += 1
            else:
                report.unjudgeable += 1
    _check_reproduction(directory, report, rows, seed)
    return report


def _is_ambiguous(observation: Observation) -> bool:
    """§11.9's trigger, set by the scoring contract and by nothing here."""
    from sweepeval.judge.escalate import AMBIGUOUS_PREFIX

    return observation.verdict is Verdict.UNSCORABLE and (
        observation.reason or ""
    ).startswith(AMBIGUOUS_PREFIX)


def is_escalatable(observation: Observation) -> bool:
    """An ambiguity a judge could be asked about, as far as the row can say.

    The `ambiguous:` marker is §11.9's trigger and is set by the scoring
    contract, not by this module. The `blob_ids` half matters as much: an
    ambiguity whose response text is gone cannot be judged offline at any
    price, and counting it would promise a resolution nothing can deliver.

    Naming a blob is not the same as that blob being readable, and this
    predicate can only see the row. Callers that are going to *spend* on the
    answer pair it with :func:`text_of`, which asks the store.
    """
    return _is_ambiguous(observation) and bool(observation.blob_ids)


def text_of(observation: Observation, store: Store) -> str | None:
    """The response text behind an observation, or ``None`` if it is gone.

    A blob id that resolves to nothing is worse than no blob id: it says the
    evidence is there. Counting those as judgeable put a number of judge calls
    in front of a user that the run could not actually spend.
    """
    for blob_id in observation.blob_ids:
        try:
            return store.blobs.get(blob_id).decode("utf-8")
        except (KeyError, UnicodeDecodeError):
            continue
    return None


def units_for(profile: str) -> dict[str, Unit]:
    """The corpus units of a profile, by id -- what a stored run was built on."""
    return {
        unit.unit_id: unit
        for template in load_corpus(profile=profile).probes
        if (unit := template.to_unit())
    }


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

    if row.get("scorer") == JUDGE_SCORER:
        # A judge verdict is not something a deterministic scorer can
        # reproduce -- that it could not is exactly why the judge was asked.
        # Re-scoring these reported every one of them as a changed verdict:
        # on the judged 14-model run that was 420 spurious changes, and it
        # marked all fourteen configs as touched, which left the reproduction
        # check with nothing to check.
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

    # The scorer is handed text, not addresses, so the observation it builds
    # names no blob. Carrying the stored ones forward keeps the re-scored row
    # pointing at the very bytes it was scored from -- without this, a
    # re-scored row is a verdict with no evidence behind it, and anything
    # persisted from `rows` (a judged run) could not be re-scored again.
    now = fresh[0].model_copy(
        update={
            "blob_ids": tuple(row.get("blob_ids") or ()),
            "call_ids": tuple(row.get("call_ids") or ()),
        }
    )
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
