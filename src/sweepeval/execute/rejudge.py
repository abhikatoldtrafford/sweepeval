"""Judge a stored run's ambiguities without re-running it (§11.9, §5.1).

The judge resolves the band a deterministic contract declared it could not
settle. Until now the only way to use it was to pay for the whole run again
with ``--judge`` set, which is the wrong price: the judge decides from the
response text, and the response text is already in the store.

On the published 14-model scorecard run that is the difference between
re-executing 7,917 target requests and spending 420 judge calls -- every
ambiguous guardrail observation in it, all fourteen models, with the target
never contacted.

**It writes a new run rather than editing the old one.** Two reasons, and both
are load-bearing. `observations.jsonl` is append-only and is what was paid for
(I7). And ``judge`` is a *hard* comparability key (§6.5): a judged result and
an unjudged one are different measurements, so `sweepeval compare` must refuse
to put them side by side -- which it does, correctly, once they are two runs.
The derived run records what it came from, and carries the blobs, so it can be
re-reported and re-scored on its own.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from sweepeval.execute.aggregation import aggregate_config
from sweepeval.execute.rescore import (
    RescoreReport,
    is_escalatable,
    rescore_run,
    text_of,
    units_for,
)
from sweepeval.http.client import TransportClient
from sweepeval.http.governor import Governor
from sweepeval.judge.client import JudgeConfig, check_independence
from sweepeval.judge.escalate import plan_escalations
from sweepeval.judge.run import aresolve
from sweepeval.schema.metric import MetricValue
from sweepeval.schema.observation import Observation
from sweepeval.scorers.base import ScorerRegistry
from sweepeval.store.derived import write_derived
from sweepeval.store.json_io import write_json
from sweepeval.store.redaction import Redactor
from sweepeval.store.run import Store, new_run_id

__all__ = ["RejudgeResult", "arejudge_run", "rejudge_run"]


@dataclass
class RejudgeResult:
    """What the judge was asked, what it answered, and where that now lives."""

    source_run_id: str
    run_id: str
    run_dir: Path
    profile: str

    escalations: int = 0
    resolved: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)
    """``(unit_id, reason)`` per call that errored or returned unusable JSON.

    A judge that fails is not a judge that passed: those rows keep their
    deterministic UNSCORABLE, and the count is reported rather than absorbed.
    """

    skipped: list[tuple[str, str]] = field(default_factory=list)
    judge_requests: int = 0
    judge_tokens_out: int = 0
    warning: str | None = None

    rescore: RescoreReport | None = None
    before: dict[str, dict[str, MetricValue]] = field(default_factory=dict)
    after: dict[str, dict[str, MetricValue]] = field(default_factory=dict)
    coverage_before: dict[str, dict[str, Any]] = field(default_factory=dict)
    coverage_after: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def unresolved(self) -> int:
        return self.escalations - self.resolved


def rejudge_run(run_dir: Path | str, **kwargs: Any) -> RejudgeResult:
    """Blocking twin of :func:`arejudge_run`."""
    import asyncio

    return asyncio.run(arejudge_run(run_dir, **kwargs))


async def arejudge_run(
    run_dir: Path | str,
    *,
    judge: JudgeConfig,
    root: str | Path = ".sweepeval",
    seed: int = 0,
    registry: ScorerRegistry | None = None,
    client: httpx.AsyncClient | None = None,
) -> RejudgeResult:
    """Escalate a stored run's ambiguities and write the judged run."""
    source = Path(run_dir)
    manifest = _manifest_of(source)
    plan = _plan_of(source)
    profile = plan.get("profile") or manifest.get("comparability", {}).get(
        "hard", {}
    ).get("profile") or "standard"

    # §11.9's one refusal, and it has to happen here rather than at the
    # endpoint: the models under test are recorded in the plan, and a judge
    # that is one of them would be scoring its own output.
    warning = check_independence(
        judge,
        (manifest.get("target") or {}).get("url", ""),
        _swept_models(plan),
    )

    report = rescore_run(source, seed=seed, registry=registry)
    units = list(units_for(profile).values())

    redactor = Redactor(secrets=[judge.key] if judge.key else [])
    source_store = Store(source.parent.parent, source.name, Redactor())
    run_id = new_run_id()
    store = Store(Path(root), run_id, redactor)

    result = RejudgeResult(
        source_run_id=manifest.get("run_id") or source.name,
        run_id=run_id,
        run_dir=store.run_dir,
        profile=profile,
        warning=warning,
        rescore=report,
    )

    # Self-contained: the judged rows name blobs, and a derived run whose
    # evidence lives only in another directory cannot be re-scored or read on
    # its own. 8MB on the scorecard run.
    _copy_blobs(source, store)

    owned = client is None
    http = client or httpx.AsyncClient(timeout=60.0, follow_redirects=True)
    transport = TransportClient(
        http, Governor(seed=seed), run_id=run_id, shape="openai.chat_completions"
    )

    judged: dict[str, list[Observation]] = {}
    try:
        for config_id, rows in sorted(report.rows.items()):
            result.before[config_id] = dict(report.metrics.get(config_id, {}))
            result.coverage_before[config_id] = _coverage(rows)

            texts = _texts_for(rows, source_store)
            plan_out = plan_escalations(rows, units, texts=texts)
            result.skipped.extend(plan_out.skipped)
            result.escalations += len(plan_out.escalations)

            extra: list[Observation] = []
            if plan_out.escalations:
                outcome = await aresolve(
                    plan_out.escalations, judge=judge, client=transport,
                    texts=texts, config_id=config_id,
                )
                result.resolved += outcome.resolved
                result.failures.extend(outcome.failures)
                result.judge_requests += outcome.requests
                result.judge_tokens_out += outcome.tokens_out
                extra = list(outcome.observations)

                if outcome.calls:
                    # §11.9: judge calls are billed and auditable like any
                    # other, and this run's calls.jsonl is exactly them.
                    store.calls.append_many([r.call for r in outcome.calls])
                    for call_result in outcome.calls:
                        if call_result.text or call_result.raw_body:
                            store.blobs.put_text(
                                call_result.text
                                or call_result.raw_body.decode("utf-8", errors="replace")
                            )

            # Appended, never substituted: the deterministic UNSCORABLE stays
            # in the log beside the judge's answer (I7).
            judged[config_id] = list(rows) + extra
            store.observations.append_many(judged[config_id])
            result.after[config_id] = dict(
                aggregate_config(
                    judged[config_id], config_id=config_id, seed=seed
                ).metrics
            )
            result.coverage_after[config_id] = _coverage(judged[config_id])
    finally:
        if owned:
            await http.aclose()

    _write_artifacts(source, store, judged, result, judge, seed=seed)
    return result


# --- reading the source ------------------------------------------------------


def _manifest_of(source: Path) -> dict[str, Any]:
    path = source / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no manifest.json in {source} — a judged re-score needs to know "
            "what the run measured, and only `sweepeval sweep` writes one."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("payload") or payload


def _plan_of(source: Path) -> dict[str, Any]:
    path = source / "plan.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("payload") or payload


def _swept_models(plan: dict[str, Any]) -> list[str]:
    """Every model this run measured, for the independence check.

    Empty when the run swept no model axis, which `check_independence` treats
    as "unknown" and refuses at a shared endpoint -- the right answer, since a
    judge that might be scoring its own output is worth nothing.
    """
    return [
        str(model)
        for config in plan.get("configs", [])
        if (model := (config.get("params") or {}).get("model"))
    ]


def _texts_for(rows: list[Observation], store: Store) -> dict[tuple[str, int], str]:
    """The response text behind each ambiguity, out of the blob store.

    Only the escalatable rows are read. The judge is shown the bytes the
    deterministic scorer saw, which is the whole basis for calling this the
    same measurement of a different kind rather than a new one.
    """
    texts: dict[tuple[str, int], str] = {}
    for row in rows:
        if not is_escalatable(row):
            continue
        text = text_of(row, store)
        if text is not None:
            texts[(row.unit_id, row.run_idx)] = text
    return texts


def _coverage(observations: list[Observation]) -> dict[str, list[int]]:
    from sweepeval.rank.coverage import count_coverage

    return {
        family: list(pair)
        for family, pair in sorted(count_coverage(observations).items())
    }


def _copy_blobs(source: Path, store: Store) -> None:
    blobs = source / "blobs"
    if blobs.is_dir():
        shutil.copytree(blobs, store.run_dir / "blobs", dirs_exist_ok=True)


# --- writing the derived run -------------------------------------------------


def _write_artifacts(
    source: Path,
    store: Store,
    judged: dict[str, list[Observation]],
    result: RejudgeResult,
    judge: JudgeConfig,
    *,
    seed: int,
) -> None:
    """`manifest.json`, `plan.json` and `aggregates.json` for the judged run.

    `aggregates.json` is derived from the source's rather than rebuilt: the
    schema is large, every reader depends on it, and the only fields a judged
    re-score changes are the per-config metrics, clusters, strata and
    coverage. Reconstructing the rest from scratch would be a second
    implementation of a format that already exists, differing from it in
    whichever field was forgotten.
    """
    manifest = _manifest_of(source)
    comparability = manifest.setdefault("comparability", {})
    comparability.setdefault("hard", {})["judge"] = judge.manifest()
    manifest["run_id"] = store.run_id
    manifest["derived_from"] = {
        "run_id": result.source_run_id,
        "run_dir": str(source),
        "how": (
            "judged re-score (§11.9): the stored responses were re-scored and "
            "the ambiguous ones escalated to an LLM judge. No target request "
            "was made; calls.jsonl holds the judge's calls and nothing else."
        ),
    }
    # `write_json`, not `write_derived`: a manifest is read unwrapped by
    # `compare` and by every reader of a run directory, and wrapping it in a
    # provenance envelope made the judged run look like a run with no
    # comparability keys at all. Its provenance is the `derived_from` block
    # above, which is part of the manifest rather than a wrapper around it.
    write_json(store.manifest_path, manifest)

    source_plan = source / "plan.json"
    if source_plan.is_file():
        shutil.copyfile(source_plan, store.plan_path)

    aggregates = source / "aggregates.json"
    if not aggregates.is_file():
        return
    payload = json.loads(aggregates.read_text(encoding="utf-8"))
    payload = payload.get("payload") or payload
    payload["run_id"] = store.run_id

    for config in payload.get("configs", []):
        config_id = config.get("config_id")
        rows = judged.get(config_id)
        if rows is None:
            continue
        fresh = aggregate_config(rows, config_id=config_id, seed=seed)
        config["metrics"] = {
            name: value.model_dump(mode="json")
            for name, value in sorted(fresh.metrics.items())
        }
        config["clusters"] = {
            metric: dict(sorted(values.items()))
            for metric, values in sorted(fresh.clusters.items())
        }
        config["strata"] = {
            metric: dict(sorted(values.items()))
            for metric, values in sorted(fresh.strata.items())
        }
        config["coverage"] = _coverage(rows)

    write_derived(
        store.aggregates_path, payload, derived_from=store.provenance()
    )
