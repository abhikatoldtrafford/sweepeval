"""The execution runner (spec §12.5, §11.7).

Executes a frozen probe set against one configuration, N times, writing calls
and observations as it goes.

Checkpointing is per ``(config_id, unit_id, run_idx)``, not per config. A
standard config is roughly 490 calls; losing all of it because a crash landed
at 99% is unacceptable on work the user paid for. Aggregation then reads only
completed unit-runs, which is how orphan rows are excluded without ever
rewriting the append-only log (I7).

Turns are **scripted**. A turn whose content depended on the target's previous
answer would unfreeze the probe set and violate I4.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sweepeval.capabilities.detect import CapabilityReport
from sweepeval.discovery.extract import extract_at
from sweepeval.discovery.ladder import LadderResult, body_for_turns
from sweepeval.http.client import TransportClient
from sweepeval.schema.call import Call, ErrorClass
from sweepeval.schema.observation import Observation
from sweepeval.schema.unit import Unit
from sweepeval.scorers import RunEvidence, ScoreContext, ScorerRegistry
from sweepeval.scorers.canary import canary_for
from sweepeval.scorers.refusal import excludes_the_trial, looks_like_refusal
from sweepeval.store.run import Store

__all__ = ["MAX_CONVERSATION_RESTARTS", "RunPlan", "UnitOutcome", "execute_config"]

MAX_CONVERSATION_RESTARTS = 2
"""§11.7. A mid-conversation failure restarts from turn 1 — resuming would
diverge state on a server session — but not forever."""


@dataclass
class RunPlan:
    """Everything one config's execution needs, frozen before it starts."""

    config_id: str
    units: tuple[Unit, ...]
    runs: int
    master_seed: str
    params: dict[str, Any] = field(default_factory=dict)
    system_prompt: str | None = None
    """Prepended as a ``system`` turn on every unit, when the config has one.

    Held on the plan rather than baked into the Units, because the Units are
    what I4 requires to be identical across configs. A system prompt written
    into a Unit would change its ``unit_id`` and make two configs of the same
    sweep incomparable by construction.
    """

    headers: dict[str, str] = field(default_factory=dict)
    """Per-config request headers, from a declared ``headers.*`` axis (§4.1).

    Kept off ``params``, which the shape passes into the request *body*: a
    routing header sweept as a body field would be sent somewhere the target
    never reads, and the axis would produce identical configs while looking
    like it swept something.
    """

    layer: str = "generic"

    def canary_table(self) -> dict[tuple[str, int, str], str]:
        """Frozen canary values (§6.1, §11.2).

        Identical across configs within a run — which is what I4 requires —
        and different between runs, so a cached response cannot pass by
        replaying an old canary.
        """
        table: dict[tuple[str, int, str], str] = {}
        for unit in self.units:
            for run_idx in range(self.runs):
                for name in unit.canary_names:
                    table[(unit.unit_id, run_idx, name)] = canary_for(
                        self.master_seed, unit.unit_id, run_idx, name
                    )
        return table


@dataclass
class UnitOutcome:
    unit: Unit
    run_idx: int
    calls: list[Call]
    observations: list[Observation]
    text: str
    restarts: int = 0
    failed: bool = False
    reason: str = ""


async def execute_config(
    client: TransportClient,
    plan: RunPlan,
    ladder: LadderResult,
    *,
    text_path: str | None,
    store: Store,
    registry: ScorerRegistry,
    capabilities: CapabilityReport | None = None,
    key: str | None = None,
    resume: bool = True,
) -> list[UnitOutcome]:
    """Run every unit N times, appending as it goes."""
    from sweepeval.discovery.auth import apply_auth

    headers, params = apply_auth(ladder.auth, key)
    headers = {**headers, **plan.headers}
    state = store.state_for(plan.config_id)
    canaries = plan.canary_table()
    outcomes: list[UnitOutcome] = []

    # A resumed config's completed unit-runs are read back off the log rather
    # than skipped into nothing. Skipping them returned no outcomes at all, so
    # aggregation -- which reads only in-memory outcomes -- saw an empty run:
    # every metric came back NO_VALID_INTERVAL, coverage 0/0, the frontier
    # degenerated, and the status was still COMPLETE. The measurements were
    # sitting in observations.jsonl the whole time, already paid for.
    restored, restored_cross_run = _restore(store, plan) if resume else ({}, [])

    for unit in plan.units:
        for run_idx in range(plan.runs):
            if resume and state.is_complete(plan.config_id, unit.unit_id, run_idx):
                previous = restored.get((unit.unit_id, run_idx))
                if previous is not None:
                    outcomes.append(previous)
                continue

            outcome = await _run_unit(
                client, plan, ladder, unit, run_idx, canaries,
                headers=headers, params=params, text_path=text_path,
                registry=registry,
            )

            store.calls.append_many(outcome.calls)
            store.observations.append_many(outcome.observations)
            if outcome.text:
                store.blobs.put_text(outcome.text)

            # Marked complete only after everything is durably appended, so a
            # crash between the two leaves orphan rows that aggregation skips
            # rather than a checkpoint that claims work which was never stored.
            state.mark_complete(plan.config_id, unit.unit_id, run_idx)
            state.record_budget(
                requests=len(outcome.calls),
                tokens=sum(c.tokens.out or 0 for c in outcome.calls),
            )
            outcomes.append(outcome)

    # Cross-run scorers need the response TEXT of every run at once. Restored
    # outcomes now carry theirs, recovered from the blob store, so the metrics
    # are recomputed correctly whether the resume was total or partial.
    #
    # The earlier guard keyed on `not executed`, which is true only for a
    # resume with nothing left to do. A resume that finished a half-run config
    # took the recompute path with `text=""` on every restored run and scored
    # determinism on empty strings -- the exact hazard the guard was written
    # for, in the one case it did not cover.
    missing = [
        o for o in outcomes
        if o.run_idx >= 0 and not o.text and not o.failed
    ]
    if missing and restored_cross_run:
        # The bytes are gone (--no-store-bodies, or over the blob cap) but the
        # previous run's verdicts are not. They were computed from the same
        # runs, so they are the answer; recomputing would be strictly worse.
        outcomes.append(
            UnitOutcome(
                unit=plan.units[0], run_idx=-1, calls=[],
                observations=restored_cross_run, text="",
                reason="cross-run observations restored from the log",
            )
        )
        return outcomes

    finalized = _finalize_cross_run(plan, outcomes, registry)
    if finalized:
        # Only when the log does not already hold them. Recomputing from the
        # same inputs gives the same rows, and appending a second identical
        # set makes an append-only log say a thing twice.
        if not restored_cross_run:
            store.observations.append_many(finalized)
        outcomes.append(
            UnitOutcome(
                unit=plan.units[0], run_idx=-1, calls=[],
                observations=finalized, text="",
                reason="cross-run observations",
            )
        )

    return outcomes


def _restore(
    store: Store, plan: RunPlan
) -> tuple[dict[tuple[str, int], UnitOutcome], list[Observation]]:
    """Rebuild outcomes for unit-runs a previous invocation already stored.

    Calls and observations both, because the operational metrics and the cache
    detector read calls while every other family reads observations. Nothing
    is re-sent: this is the log, not the target.
    """
    units = {unit.unit_id: unit for unit in plan.units}
    calls: dict[tuple[str, int], list[Call]] = {}
    observations: dict[tuple[str, int], list[Observation]] = {}
    cross_run: list[Observation] = []

    for call in store.calls.read():
        if call.config_id != plan.config_id or call.unit_id not in units:
            continue
        calls.setdefault((call.unit_id, call.run_idx), []).append(call)

    # A cross-run scorer's rows describe the SET of runs, so they belong to no
    # single unit-run and are restored as a group.
    cross_run_metrics = {
        "config_repeatability",
        "target_determinism_at_temp0",
        "semantic_stability",
        "invariance",
    }
    for observation in store.observations.read():
        if observation.config_id != plan.config_id:
            continue
        if observation.metric in cross_run_metrics:
            cross_run.append(observation)
            continue
        if observation.unit_id not in units:
            continue
        observations.setdefault(
            (observation.unit_id, observation.run_idx), []
        ).append(observation)

    return {
        key: UnitOutcome(
            unit=units[key[0]],
            run_idx=key[1],
            calls=calls.get(key, []),
            observations=rows,
            # Recovered from the blob store, not left empty. A restored
            # outcome with `text=""` is what let a PARTIAL resume score
            # determinism on empty strings: `_finalize_cross_run` marks a
            # textless outcome unscorable, so a Ctrl-C plus --resume turned
            # `target_determinism_at_temp0` 1.00 into 0.00 -- with an
            # interval, labelled COMPLETE -- and propagated it to every
            # temperature sibling through `_share_determinism`.
            text=_restore_text(store, rows),
            reason="restored from the log by --resume",
        )
        for key, rows in observations.items()
    }, cross_run


def _restore_text(store: Store, rows: Sequence[Observation]) -> str:
    """The response text a stored observation was scored from.

    Every observation defaults its ``blob_ids`` to the address of the text it
    scored, precisely so a stored run can be re-scored without paying again.
    Returns ``""`` when the bytes are genuinely gone -- ``--no-store-bodies``,
    or text over the blob cap -- and the caller must then decline the
    cross-run metrics rather than compute them on nothing.
    """
    for row in rows:
        for blob_id in row.blob_ids:
            try:
                return store.blobs.get(blob_id).decode("utf-8")
            except (KeyError, UnicodeDecodeError):
                continue
    return ""


def _finalize_cross_run(
    plan: RunPlan, outcomes: Sequence[UnitOutcome], registry: ScorerRegistry
) -> list[Observation]:
    """Give cross-run scorers every run at once (§11.4).

    Determinism cannot work one run at a time: repeatability, semantic
    stability and invariance are statements about the *set* of runs. Rather
    than let a scorer smuggle state between per-run calls, the runner collects
    the evidence and hands it over in one go.
    """
    by_unit: dict[str, dict[int, str]] = {}
    unscorable: dict[str, set[int]] = {}
    units: dict[str, Unit] = {}

    for outcome in outcomes:
        if outcome.run_idx < 0:
            continue
        units[outcome.unit.unit_id] = outcome.unit
        by_unit.setdefault(outcome.unit.unit_id, {})[outcome.run_idx] = outcome.text
        # §11.8: a refusal on a determinism unit excludes the trial. Without
        # this the metric is a statement about how consistently the target
        # declines -- three identical refusals score 1.0, so the objective is
        # maximised by a target that answers nothing.
        refused = excludes_the_trial(outcome.unit) and looks_like_refusal(outcome.text)
        if outcome.failed or not outcome.text or refused:
            unscorable.setdefault(outcome.unit.unit_id, set()).add(outcome.run_idx)

    evidence = [
        RunEvidence(
            unit=units[unit_id],
            texts=tuple(runs.get(i, "") for i in range(plan.runs)),
            unscorable=tuple(sorted(unscorable.get(unit_id, set()))),
        )
        for unit_id, runs in sorted(by_unit.items())
    ]
    if not evidence:
        return []

    context = ScoreContext(
        run_id=plan.config_id, config_id=plan.config_id, run_idx=0, text="",
        layer=plan.layer, ts=datetime.now(timezone.utc).isoformat(),
    )

    observations: list[Observation] = []
    for scorer in registry.all():
        finalize = getattr(scorer, "finalize", None)
        if callable(finalize):
            observations.extend(finalize(evidence, context))
    return observations


async def _run_unit(
    client: TransportClient,
    plan: RunPlan,
    ladder: LadderResult,
    unit: Unit,
    run_idx: int,
    canaries: Mapping[tuple[str, int, str], str],
    *,
    headers: dict[str, str],
    params: dict[str, Any],
    text_path: str | None,
    registry: ScorerRegistry,
) -> UnitOutcome:
    unit_canaries = {
        name: canaries[(unit.unit_id, run_idx, name)] for name in unit.canary_names
    }

    calls: list[Call] = []
    text = ""
    restarts = 0
    failed = False
    reason = ""

    while True:
        calls, text, ok = await _play_conversation(
            client, plan, ladder, unit, run_idx, unit_canaries,
            headers=headers, params=params, text_path=text_path,
        )
        if ok:
            break
        restarts += 1
        if restarts > MAX_CONVERSATION_RESTARTS:
            failed = True
            reason = "conversation_failed"
            break

    observations = _score(
        unit, calls, text, unit_canaries, plan, run_idx, registry,
        failed=failed, reason=reason,
    )
    return UnitOutcome(
        unit=unit, run_idx=run_idx, calls=calls, observations=observations,
        text=text, restarts=restarts, failed=failed, reason=reason,
    )


async def _play_conversation(
    client: TransportClient,
    plan: RunPlan,
    ladder: LadderResult,
    unit: Unit,
    run_idx: int,
    unit_canaries: Mapping[str, str],
    *,
    headers: dict[str, str],
    params: dict[str, Any],
    text_path: str | None,
) -> tuple[list[Call], str, bool]:
    """Play a unit's scripted turns by stateless replay (§9.2)."""
    history: list[tuple[str, str]] = []
    if plan.system_prompt:
        history.append(("system", plan.system_prompt))
    calls: list[Call] = []
    text = ""

    for turn_idx, turn in enumerate(unit.turns):
        rendered = _render(turn.text, unit_canaries)
        history.append((turn.role, rendered))

        # A system turn is context, not a request. Posting after one sends a
        # conversation with no user message in it, spends a call, and then
        # appends the model's reply to that non-question as an assistant turn
        # -- so the probe the unit actually wanted to send arrives with a
        # fabricated exchange already in front of it.
        if turn.role != "user":
            continue

        body = body_for_turns(ladder, list(history), **plan.params)
        results = await client.call(
            ladder.path, body,
            config_id=plan.config_id, unit_id=unit.unit_id,
            run_idx=run_idx, turn_idx=turn_idx,
            headers=headers, params=params,
        )
        calls.extend(r.call for r in results)

        final = results[-1]
        if final.call.response.error_class is not ErrorClass.ok:
            # §11.7: restart the conversation rather than resume mid-way,
            # which would diverge state on a server session.
            return calls, text, False

        text = _extract(final.raw_body, text_path)
        history.append(("assistant", text))

    return calls, text, True


def _render(text: str, canaries: Mapping[str, str]) -> str:
    out = text
    for name, value in canaries.items():
        out = out.replace(f"{{{{canary:{name}}}}}", value)
        if name == "primary":
            out = out.replace("{{canary}}", value)
    return out


def _extract(raw: bytes, text_path: str | None) -> str:
    import json

    if not raw or not text_path:
        return ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    return extract_at(payload, text_path) or ""


def _score(
    unit: Unit,
    calls: Sequence[Call],
    text: str,
    unit_canaries: Mapping[str, str],
    plan: RunPlan,
    run_idx: int,
    registry: ScorerRegistry,
    *,
    failed: bool,
    reason: str,
) -> list[Observation]:
    from sweepeval.schema.hashing import sha256_hex
    from sweepeval.schema.observation import Verdict

    context = ScoreContext(
        run_id=calls[0].run_id if calls else "",
        config_id=plan.config_id,
        run_idx=run_idx,
        text=text,
        # The same address the blob store will file this text under, so the
        # observation and the bytes it scored can be rejoined later.
        text_blob_id=sha256_hex(text.encode("utf-8")) if text else None,
        canaries=dict(unit_canaries),
        refusal_detected=looks_like_refusal(text),
        layer=plan.layer,
        ts=datetime.now(timezone.utc).isoformat(),
    )

    # Operational metrics derive from calls.jsonl and therefore from EVERY
    # unit, not from a family of its own (§6.2). Scoring them only for units
    # whose family is "operational" left latency, tokens and error_rate with
    # no data at all — and two of the six default objectives, latency_p95_ms
    # and cost_per_probe, silently measuring nothing.
    observations: list[Observation] = list(
        _operational(unit, calls, context, registry)
    )

    if failed:
        # I5: a failed conversation is UNSCORABLE with a reason, never a zero.
        observations.append(
            context.observation(
                scorer=unit.family, version=0, metric=f"{unit.family}_pass_rate",
                family=unit.family, verdict=Verdict.UNSCORABLE,
                reason=reason, unit=unit,
            )
        )
        return observations

    try:
        scorer = registry.get(unit.family)
    except KeyError:
        return observations
    observations.extend(scorer.score(unit, list(calls), context))
    return observations


def _operational(
    unit: Unit,
    calls: Sequence[Call],
    context: ScoreContext,
    registry: ScorerRegistry,
) -> list[Observation]:
    """Latency, tokens and error rate for one unit-run.

    Skipped when the unit is itself operational, which would score it twice,
    and when there are no calls, because an empty conversation has no timing
    to report and a zero would read as an instant answer.
    """
    if unit.family == "operational" or not calls:
        return []
    try:
        scorer = registry.get("operational")
    except KeyError:
        return []
    return list(scorer.score(unit, list(calls), context))


