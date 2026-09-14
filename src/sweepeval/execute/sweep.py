"""The sweep runner (spec §12, D15, D22). I4 and I9 load-bearing.

Discover once, detect once, plan once, then execute every configuration
against the **same frozen probe set** with the **same canary table**. That
sharing is not an optimisation; it is what makes the paired test legitimate.
Two configs scored on different probes cannot be compared by pairing on the
probe, and every interval downstream assumes the pairing holds (I4).

Three things this module refuses to do, each because an earlier revision did:

* **Spend before consenting.** The estimate covers discovery and capability
  detection too, and it is shown before the first request of any kind, using
  the profile's cap as the config count because the real count is not knowable
  until discovery has already spent (I9).
* **Rank.** The layering contract forbids importing ``rank.domination`` here,
  so a sweep produces per-config aggregates and nothing that resembles an
  ordering. Ranking is M9's, through one module that goes via Holm.
* **Half-finish silently.** At the budget cap the sweep stops between configs,
  reports ``INCOMPLETE``, and names every config that never ran. A frontier
  over a subset presented as a frontier over the sweep is a lie the artifact
  cannot detect later.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import httpx

from sweepeval.capabilities.detect import (
    Capability,
    CapabilityBudget,
    CapabilityReport,
    detect_all,
)
from sweepeval.capabilities.sampling_probe import SamplingReport, probe_sampling
from sweepeval.corpus.loader import Corpus, load_corpus
from sweepeval.corpus.template import Profile
from sweepeval.discovery.budget import DiscoveryBudget
from sweepeval.discovery.runner import DiscoveryOutcome, discover_target
from sweepeval.execute.aggregation import ConfigAggregate, aggregate_config
from sweepeval.execute.artifacts import (
    ResumeVerdict,
    build_manifest,
    build_plan_document,
    read_json,
    verify_resume,
    write_json,
)
from sweepeval.execute.authz import (
    AuthorizationRecord,
    AuthorizationStore,
    require_authorization,
)
from sweepeval.execute.budget import BudgetCap, Estimate, estimate_run
from sweepeval.execute.cache_detect import CacheVerdict, detect_cache, flag_affected
from sweepeval.execute.cost import CostAccounting, Pricing, account
from sweepeval.execute.declared import DeclaredConfig
from sweepeval.execute.hard_fail import HardFailReport, classify_hard_fails
from sweepeval.execute.planner import CAP_BY_PROFILE, ConfigSpec, SweepPlan, plan_sweep
from sweepeval.execute.runner import RunPlan, UnitOutcome, execute_config
from sweepeval.http.client import TransportClient
from sweepeval.http.governor import DEFAULT_CONCURRENCY, Governor
from sweepeval.judge.client import JudgeConfig, check_independence
from sweepeval.schema.call import Call
from sweepeval.schema.comparability import (
    Comparability,
    HardKeys,
    JudgeKey,
    SoftKeys,
)
from sweepeval.schema.observation import Observation
from sweepeval.schema.unit import Unit
from sweepeval.schema.versions import SCHEMA_MAJOR, SUITE_VERSION, TOOL_VERSION
from sweepeval.scorers import registry as scorer_registry
from sweepeval.store.redaction import Redactor
from sweepeval.store.run import Store, new_run_id

__all__ = [
    "ConfigResult",
    "SweepResult",
    "SweepStatus",
    "asweep_target",
]

ConfirmFn = Callable[[Estimate], bool]


class SweepStatus(str, Enum):
    """How a sweep ended (§12.3, §12.5)."""

    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    """Stopped at the budget cap; configs that never ran are named."""

    DECLINED = "DECLINED"
    """The user did not confirm the estimate. Nothing was sent (I9)."""

    REFUSED = "REFUSED"
    """``--resume`` found a changed plan, corpus or hard key (§12.5)."""


@dataclass
class ConfigResult:
    """One row of the sweep."""

    config: ConfigSpec
    outcomes: list[UnitOutcome] = field(default_factory=list)
    aggregate: ConfigAggregate | None = None
    cache: CacheVerdict = field(default_factory=CacheVerdict)
    cost: CostAccounting | None = None
    hard_fails: HardFailReport = field(default_factory=HardFailReport)
    judge: Any = None
    """The §11.9 outcome when a judge ran: what it resolved, and what it
    could not. Carried so the report can say how many of this row's
    numbers a model decided rather than a contract."""

    requests: int = 0

    @property
    def config_id(self) -> str:
        return self.config.config_id

    @property
    def observations(self) -> list[Observation]:
        return [o for outcome in self.outcomes for o in outcome.observations]

    @property
    def calls(self) -> list[Call]:
        return [c for outcome in self.outcomes for c in outcome.calls]

    @property
    def metrics(self) -> dict[str, Any]:
        return dict(self.aggregate.metrics) if self.aggregate else {}

    @property
    def clusters(self) -> dict[str, dict[str, float]]:
        return dict(self.aggregate.clusters) if self.aggregate else {}

    @property
    def tokens_out(self) -> int:
        return sum(c.tokens.out or 0 for c in self.calls)

    @property
    def tokens_in(self) -> int:
        return sum(c.tokens.in_ or 0 for c in self.calls)

    @property
    def strata(self) -> dict[str, dict[str, str]]:
        return dict(self.aggregate.strata) if self.aggregate else {}

    @property
    def coverage(self) -> dict[str, tuple[int, int]]:
        """``family -> (scored, attempted)``, for §14.5's parity check."""
        from sweepeval.rank.coverage import count_coverage

        return count_coverage(self.observations)


@dataclass
class SweepResult:
    """Everything a report or a ranker needs, and nothing ranked."""

    run_id: str
    profile: Profile
    runs: int
    status: SweepStatus
    corpus: Corpus
    plan: SweepPlan
    discovery: DiscoveryOutcome | None = None
    capabilities: CapabilityReport = field(default_factory=CapabilityReport)
    comparability: Comparability | None = None
    authorization: AuthorizationRecord | None = None
    estimate: Estimate | None = None
    configs: list[ConfigResult] = field(default_factory=list)
    not_run: list[str] = field(default_factory=list)
    """Configs the budget cap stopped before. Named, never dropped."""

    families_not_run: tuple[str, ...] = ()
    skipped: list[tuple[str, str]] = field(default_factory=list)
    assumptions: list[tuple[str, str, str]] = field(default_factory=list)
    sampling: SamplingReport = field(default_factory=SamplingReport)
    declared: DeclaredConfig | None = None
    """The user's overrides, when a config was supplied (§8.6)."""

    determinism_scope: dict[str, str] = field(default_factory=dict)
    """Which config each row's ``target_determinism_at_temp0`` was measured on
    (§14.2). A row pointing at itself measured its own; a row pointing
    elsewhere inherited its (model, system_prompt) sibling's temp=0 figure."""

    resume: ResumeVerdict | None = None
    stop_reason: str = ""
    store: Store | None = None
    units: tuple[Unit, ...] = ()

    @property
    def single_config(self) -> bool:
        """D15: nothing to sweep, so this is an evaluation with a banner."""
        return len(self.plan.configs) <= 1

    @property
    def complete(self) -> bool:
        return self.status is SweepStatus.COMPLETE

    @property
    def config_ids(self) -> tuple[str, ...]:
        return tuple(c.config_id for c in self.configs)

    def by_id(self, config_id: str) -> ConfigResult | None:
        return next((c for c in self.configs if c.config_id == config_id), None)

    def clusters_for(self, config_id: str, metric: str) -> dict[str, float]:
        """The matched blocks a paired comparison needs (§13.3)."""
        row = self.by_id(config_id)
        return dict(row.clusters.get(metric, {})) if row else {}


async def asweep_target(
    url: str,
    *,
    key: str | None = None,
    profile: Profile = "quick",
    runs: int = 3,
    root: str = ".sweepeval",
    seed: int = 0,
    client: httpx.AsyncClient | None = None,
    authorized: bool = False,
    authorization_prompt: bool = True,
    store_text: bool = True,
    confirm: ConfirmFn | None = None,
    cap: BudgetCap | None = None,
    pricing: Pricing | None = None,
    resume_run_id: str | None = None,
    config_cap: int | None = None,
    declared: DeclaredConfig | None = None,
    judge: JudgeConfig | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> SweepResult:
    """Discover, plan, sweep and aggregate. Ranking is the caller's.

    ``confirm`` receives the pre-flight estimate and returns whether to spend.
    ``None`` means non-interactive consent — the CLI always passes one, and a
    library caller who omits it has said yes by calling the function.
    """
    say = on_progress or (lambda _msg: None)
    budget_cap = cap or BudgetCap()

    # I9: the estimate precedes every request, so it is built from the
    # profile's cap rather than from a config count discovery has not produced
    # yet. Planning can only shrink the sweep, never grow it past the cap, so
    # the figure shown is an upper bound the run cannot exceed.
    corpus = load_corpus(profile)
    # The cap the estimate is built from is the one the planner will use, so
    # the figure shown is an upper bound the run cannot exceed.
    # An explicit config_cap overrides the profile default in BOTH directions.
    # Lowering it is a cheaper run; raising it is a deliberate, larger one --
    # comparing eight models needs eight configs, and the profile cap of six
    # would silently truncate two of them out of the scorecard. The estimate
    # below is computed from whatever this ends up being, so the user still
    # consents to the real figure.
    if judge is not None and declared is not None:
        # §11.9, before the pre-flight where the models are already declared:
        # a judge that is one of them scores its own output. Raised here so a
        # misconfiguration costs nothing to discover.
        check_independence(judge, url, _declared_models(declared))

    planned_cap = config_cap or CAP_BY_PROFILE[profile]
    estimate = estimate_run(
        corpus,
        configs=planned_cap,
        runs=runs,
        profile=profile,
        pricing_source=pricing.source if pricing else "none",
        # I9: the judge spends too, and the worst case is bounded by how many
        # probes can return AMBIGUOUS at all. Left out, a judged run's
        # estimate silently excluded thousands of requests.
        judge_units=len(corpus.ambiguity_capable) if judge else 0,
    )
    if confirm is not None and not confirm(estimate):
        return SweepResult(
            run_id="",
            profile=profile,
            runs=runs,
            status=SweepStatus.DECLINED,
            corpus=corpus,
            plan=SweepPlan(configs=()),
            estimate=estimate,
            stop_reason="the estimate was not confirmed; nothing was sent",
        )
    if budget_cap.forbids_starting(estimate, pricing):
        return SweepResult(
            run_id="",
            profile=profile,
            runs=runs,
            status=SweepStatus.DECLINED,
            corpus=corpus,
            plan=SweepPlan(configs=()),
            estimate=estimate,
            stop_reason=(
                f"the {budget_cap.unit} cap of {budget_cap.value} is below the "
                f"{estimate.unavoidable_requests} request(s) discovery and "
                f"capability detection need, so no configuration could be "
                f"scored at all; nothing was sent"
            ),
        )

    owned = client is None
    http = client or httpx.AsyncClient(timeout=60.0, follow_redirects=True)
    run_id = resume_run_id or new_run_id()

    try:
        say("discovering")
        discovery = await discover_target(
            http, url, key, budget=DiscoveryBudget(), seed=seed
        )
        say(f"shape {discovery.ladder.shape.name} at {discovery.ladder.path}")

        capability_budget = CapabilityBudget()
        capabilities = await detect_all(
            http,
            discovery.ladder,
            key,
            discovery.extraction.path,
            budget=capability_budget,
            profile=profile,
        )

        # The sampling test shares the capability budget with the detectors,
        # and runs after them: whether a parameter moves the output is only
        # worth paying for once we know the target answers at all.
        sampling = await probe_sampling(
            http,
            discovery.ladder,
            key,
            discovery.extraction.path,
            [t.turns[0].text for t in corpus.instruments if t.turns],
            budget=capability_budget,
            seed=seed,
        )

        plan = plan_sweep(
            capabilities,
            list(discovery.ladder.sniff.models),
            profile=profile,
            sampling=sampling.verdicts,
            sampling_notes=sampling.not_tested,
            cap=planned_cap,
            declared_axes=declared.axes if declared else None,
        )

        # A declared extraction path replaces the inferred one. Applied here,
        # after discovery has run, because the capability detectors need a
        # path too and correcting it afterwards would leave them looking at
        # the wrong field.
        text_path = discovery.extraction.path
        if declared is not None and declared.text_path:
            text_path = declared.text_path
            say(f"extraction path overridden by config: {text_path}")
        say(
            f"{len(plan.configs)} config(s) over axes "
            f"{', '.join(sorted(plan.axes)) or '(none)'}"
        )

        authorization = require_authorization(
            url,
            store=AuthorizationStore(Path(root)),
            flag=authorized,
            prompt=authorization_prompt,
        )

        redactor = Redactor(secrets=[key] if key else [])
        store = Store(Path(root), run_id, redactor, store_text=store_text)

        # Families the capability report already ruled out are not executed.
        # Running them would spend real money producing rows for a family the
        # report calls SKIPPED, and context alone is over half the calls at
        # `standard`.
        skipped_families = {s.family for s, _ in scorer_registry().skipped(capabilities)}
        units = tuple(
            t.to_unit() for t in corpus.probes if t.family not in skipped_families
        )
        not_run_families = sorted({t.family for t in corpus.probes} & skipped_families)

        master_seed = f"{run_id}:{seed}"
        judge_warning: str | None = None
        if judge is not None:
            # Authoritative check: by here the plan exists, so every model
            # actually under test is known -- including ones discovery found
            # rather than the user declaring. Any warning is recorded as an
            # assumption the user can see and correct (§8.6).
            judge_warning = check_independence(
                judge, url, [c.params.get("model", "") for c in plan.configs]
            )

        comparability = _comparability(
            discovery, corpus, profile, runs, text_path, judge
        )

        # I4: one canary table for the whole sweep. Built once, here, and
        # handed to every config unchanged.
        canaries = RunPlan(
            config_id="_shared", units=units, runs=runs, master_seed=master_seed
        ).canary_table()

        document = build_plan_document(
            plan,
            units,
            canaries,
            run_id=run_id,
            profile=profile,
            runs=runs,
            master_seed=master_seed,
            corpus_hash=corpus.hash,
            determinism_sharing=_determinism_sharing(plan),
        )
        plan_hash = document.hash()

        resume: ResumeVerdict | None = None
        if resume_run_id:
            resume = verify_resume(
                read_json(store.manifest_path),
                comparability=comparability,
                plan_hash=plan_hash,
                corpus_hash=corpus.hash,
            )
            if not resume.ok:
                return SweepResult(
                    run_id=run_id,
                    profile=profile,
                    runs=runs,
                    status=SweepStatus.REFUSED,
                    corpus=corpus,
                    plan=plan,
                    discovery=discovery,
                    capabilities=capabilities,
                    comparability=comparability,
                    estimate=estimate,
                    resume=resume,
                    stop_reason=resume.explain(),
                    store=store,
                )

        write_json(store.plan_path, document.to_dict())
        write_json(
            store.manifest_path,
            build_manifest(
                run_id=run_id,
                comparability=comparability,
                plan_hash=plan_hash,
                corpus_hash=corpus.hash,
                capabilities=capabilities.to_manifest(),
                target={
                    "url": redactor.url(url),
                    "target_type": discovery.target_type,
                    "shape": discovery.ladder.shape.name,
                    "path": discovery.ladder.path,
                    "extraction_path": text_path or "",
                    "auth": discovery.ladder.auth.name,
                },
                authorization=authorization.to_manifest() if authorization else None,
                seed=seed,
                master_seed=master_seed,
                budget={
                    "estimated_requests": estimate.total_requests,
                    "estimated_tokens": estimate.total_tokens,
                    "cap_unit": budget_cap.unit,
                    "cap_value": budget_cap.value,
                },
                heuristics={"tokens": "chars/4", "wall_clock": "2.5s per request"},
            ),
        )

        result = SweepResult(
            run_id=run_id,
            profile=profile,
            runs=runs,
            status=SweepStatus.COMPLETE,
            corpus=corpus,
            plan=plan,
            discovery=discovery,
            capabilities=capabilities,
            comparability=comparability,
            authorization=authorization,
            estimate=estimate,
            declared=declared,
            families_not_run=tuple(not_run_families),
            sampling=sampling,
            resume=resume,
            store=store,
            units=units,
        )

        transport = TransportClient(
            http,
            Governor(seed=seed),
            run_id=run_id,
            shape=discovery.ladder.shape.name,
        )

        # Discovery and capability detection are billable and they went first.
        # Seeding the counter from per-config state alone under-counted the
        # cap by that whole phase: a 100-request cap let 157 requests through.
        spent = _already_spent(store, plan) + discovery.budget.posts
        spent += capability_budget.posts
        # Kept apart, because they are priced apart. Collapsing them into
        # one figure and passing it to `pricing.cost(0, tokens)` billed
        # every input token at the OUTPUT rate -- a 10x overstatement at
        # typical prices, in the direction that stops a run early.
        # Discovery and capability probes are inert and short; their spend
        # is counted as input, which is where nearly all of it is.
        tokens_in = discovery.budget.tokens + capability_budget.tokens
        tokens_out = 0
        for index, config in enumerate(plan.configs):
            # The cap is checked between configs, never mid-config: stopping
            # inside one leaves a config with partial coverage, and a partially
            # covered config is worse than an absent one — it can be compared
            # against, and the comparison is wrong (§14.5).
            # Projected, not reached: a hard cap that only notices after the
            # fact is not a cap. Checked between configs because stopping
            # inside one leaves partial coverage, which is worse than an
            # absent config -- it can still be compared against (§14.5).
            if _cap_reached(
                budget_cap,
                spent + estimate.per_config_requests,
                tokens_in=tokens_in + estimate.per_config_tokens,
                tokens_out=tokens_out,
                pricing=pricing,
            ):
                result.status = SweepStatus.INCOMPLETE
                result.not_run = [c.config_id for c in plan.configs[index:]]
                result.stop_reason = (
                    f"stopped before config {index + 1} of {len(plan.configs)}: "
                    f"another one would cross the {budget_cap.unit} cap of "
                    f"{budget_cap.value} (spent {spent} requests so far)"
                )
                break

            say(f"config {index + 1}/{len(plan.configs)}: {config.label()}")
            row = await _run_one(
                transport,
                config,
                units=units,
                runs=runs,
                master_seed=master_seed,
                ladder=discovery.ladder,
                text_path=text_path,
                store=store,
                key=key,
                seed=seed,
                profile=profile,
                pricing=pricing,
                judge=judge,
            )
            spent += row.requests
            tokens_in += row.tokens_in
            tokens_out += row.tokens_out
            result.configs.append(row)

        _share_determinism(result, _determinism_sharing(plan))
    finally:
        if owned:
            await http.aclose()

    _collect_skips(result)
    _collect_assumptions(result)
    if judge_warning:
        # Disclosed, not blocked: independence cannot be established from a
        # black box, so §11.9 asks for the record rather than a gate.
        result.assumptions.append(("judge.independence", "low", judge_warning))
    return result


def _share_determinism(result: SweepResult, sharing: Mapping[str, str]) -> None:
    """Apply §14.2: temp=0 determinism is shared, not re-measured per row.

    Measured at each config's own settings, ``target_determinism_at_temp0``
    would be ~1.0 for every temp=0 config and ~0 for every temp=1.0 config by
    construction — a manufactured win on a metric that merely restates the
    row's own label, and every temp=0 config would be non-dominated for free.

    Sharing relocates the structure rather than removing it: the objective
    takes one value per (model, system_prompt) pair and is exactly tied within
    each temperature triple. A tie does not block domination the way a
    manufactured win does, which is the whole point.

    ``config_repeatability`` is deliberately left alone. It is the
    production-truth number at the row's own settings, and it is reported
    beside the shared one.
    """
    metric = "target_determinism_at_temp0"
    rows = {row.config_id: row for row in result.configs}

    for config_id, owner_id in sorted(sharing.items()):
        row = rows.get(config_id)
        if row is None or row.aggregate is None:
            continue
        if owner_id == config_id:
            result.determinism_scope[config_id] = config_id
            continue

        owner = rows.get(owner_id)
        if owner is None or owner.aggregate is None:
            # The owner never ran — the budget cap stopped before it. Keep the
            # row's own measurement rather than dropping the objective, and
            # say whose it is, because a shared number attributed to a config
            # that was never executed is worse than an honest self-measurement.
            result.determinism_scope[config_id] = f"{config_id} (own settings)"
            continue

        shared = owner.aggregate.metrics.get(metric)
        if shared is None:
            result.determinism_scope[config_id] = f"{config_id} (own settings)"
            continue

        row.aggregate.metrics[metric] = shared
        # The clusters go with it: the paired test resamples them, and leaving
        # the row's own blocks beside a borrowed point estimate would compare
        # one config's number against another config's variance.
        row.aggregate.clusters[metric] = dict(
            owner.aggregate.clusters.get(metric, {})
        )
        result.determinism_scope[config_id] = owner_id


async def _run_one(
    transport: TransportClient,
    config: ConfigSpec,
    *,
    units: tuple[Unit, ...],
    runs: int,
    master_seed: str,
    ladder: Any,
    text_path: str | None,
    store: Store,
    key: str | None,
    seed: int,
    profile: Profile,
    pricing: Pricing | None,
    judge: JudgeConfig | None = None,
) -> ConfigResult:
    """Execute one configuration and aggregate it."""
    # A declared ``headers.x-...`` axis is a header, not a body field.
    headers = {
        name[len("headers.") :]: str(value)
        for name, value in config.params.items()
        if name.startswith("headers.")
    }
    params = {
        name: value
        for name, value in config.params.items()
        if not name.startswith("headers.")
    }

    run_plan = RunPlan(
        config_id=config.config_id,
        units=units,
        runs=runs,
        master_seed=master_seed,
        params=params,
        headers=headers,
        system_prompt=config.system_prompt,
    )
    outcomes = await execute_config(
        transport,
        run_plan,
        ladder,
        text_path=text_path,
        store=store,
        registry=scorer_registry(),
        key=key,
    )

    row = ConfigResult(config=config, outcomes=outcomes)

    # §11.9: resolve the ambiguities a deterministic contract declared it
    # could not settle. Between execution and aggregation, so the judge's
    # verdicts are in the observation set the metrics are computed from.
    if judge is not None:
        row.judge = await _judge_config(
            row, units=units, judge=judge, transport=transport,
            store=store, config_id=config.config_id,
        )

    row.requests = len(row.calls)
    row.cache = detect_cache(row.calls, config_id=config.config_id)
    row.cost = account(row.calls, pricing=pricing)
    row.hard_fails = classify_hard_fails(
        row.observations, config_id=config.config_id
    )

    aggregate = aggregate_config(
        row.observations,
        config_id=config.config_id,
        seed=seed,
        indicative=profile == "quick",
        # So the cost objective ranks what the user is actually billed,
        # instead of output tokens while the cost block beside it prints
        # dollars.
        pricing=pricing,
    )
    # §12.7: the flag rides on every metric a cache would corrupt, attached
    # here rather than in the aggregator, which has no view of the calls.
    aggregate.metrics = flag_affected(aggregate.metrics, row.cache)
    row.aggregate = aggregate
    return row


async def _judge_config(
    row: ConfigResult,
    *,
    units: tuple[Unit, ...],
    judge: JudgeConfig,
    transport: TransportClient,
    store: Store,
    config_id: str,
) -> Any:
    """Escalate this config's ambiguities and append the verdicts.

    The response text comes from the blob store rather than from memory, so a
    resumed run judges the same bytes the original scored -- and so this works
    at all on a config whose outcomes were restored rather than executed.
    """
    from sweepeval.judge.escalate import plan_escalations
    from sweepeval.judge.run import aresolve

    texts = {
        (outcome.unit.unit_id, outcome.run_idx): outcome.text
        for outcome in row.outcomes
        if outcome.run_idx >= 0 and outcome.text
    }
    plan = plan_escalations(row.observations, units, texts=texts)
    if not plan.escalations:
        return None

    outcome = await aresolve(
        plan.escalations, judge=judge, client=transport,
        texts=texts, config_id=config_id,
    )
    if outcome.calls:
        # §11.9: judge calls are billed and auditable like any other.
        store.calls.append_many([r.call for r in outcome.calls])
        for result in outcome.calls:
            if result.text or result.raw_body:
                store.blobs.put_text(result.text or result.raw_body.decode(
                    "utf-8", errors="replace"
                ))

    if outcome.observations:
        # Appended, never substituted: the deterministic UNSCORABLE stays in
        # the log beside the judge's answer (I7).
        store.observations.append_many(outcome.observations)
        row.outcomes.append(
            UnitOutcome(
                unit=units[0], run_idx=-1, calls=[],
                observations=outcome.observations, text="",
                reason="judge verdicts (§11.9)",
            )
        )
    return outcome


def _cap_reached(
    cap: BudgetCap,
    spent: int,
    *,
    tokens_in: int,
    tokens_out: int,
    pricing: Pricing | None,
) -> bool:
    """Whether the cap is reached, in whichever unit it was set.

    All three units bind. The token cap was never checked against a real
    count, and the dollar cap returned False unconditionally -- a
    ``BudgetCap(0.01, "dollars")`` completed a 757-request run, and a test
    enshrined that as intended.

    Input and output tokens arrive separately because they are priced
    separately. Passing their sum as the output count billed input at the
    output rate, overstating projected spend by the price ratio -- 10x at
    typical rates, in the direction that stops a run that had budget left.
    """
    if cap.value is None:
        return False
    if cap.unit == "requests":
        return spent >= cap.value
    if cap.unit == "tokens":
        return tokens_in + tokens_out >= cap.value
    if cap.unit == "dollars":
        # Without pricing there is no dollar figure to compare, and inventing
        # one is what D8 forbids. The cap cannot bind; the caller was told so
        # by the pre-flight, which says the cost objective is in tokens.
        if pricing is None:
            return False
        return pricing.cost(tokens_in, tokens_out) >= cap.value
    return False


def _already_spent(store: Store, plan: SweepPlan) -> int:
    """Reconstruct the counter across configs so a resume respects the cap."""
    total = 0
    for config in plan.configs:
        requests, _tokens = store.state_for(config.config_id).budget()
        total += requests
    return total


def _determinism_sharing(plan: SweepPlan) -> dict[str, str]:
    """Which config each temperature sibling borrows determinism from (§11.4).

    ``target_determinism_at_temp0`` is a property of the (model,
    system_prompt) pair, so it is measured once at temp=0 and shared with that
    pair's other temperatures. Written into the plan so a reader can see which
    row a number came from rather than assuming every row measured its own.
    """
    owners: dict[tuple[Any, str], str] = {}
    for config in plan.configs:
        if config.params.get("temperature", 0.0) == 0.0:
            owners.setdefault(
                (config.params.get("model"), config.system_prompt_variant),
                config.config_id,
            )

    sharing: dict[str, str] = {}
    for config in plan.configs:
        owner = owners.get(
            (config.params.get("model"), config.system_prompt_variant)
        )
        if owner:
            sharing[config.config_id] = owner
    return sharing


def _declared_models(declared: DeclaredConfig | None) -> list[str]:
    """Model ids the user declared, for the judge independence check."""
    if declared is None:
        return []
    return [str(m) for m in declared.axes.get("model", []) if m]


def _comparability(
    discovery: DiscoveryOutcome,
    corpus: Corpus,
    profile: Profile,
    runs: int,
    text_path: str | None = None,
    judge: JudgeConfig | None = None,
) -> Comparability:
    scorers = {s.family: s.version for s in scorer_registry().all()}
    return Comparability(
        hard=HardKeys(
            schema_major=SCHEMA_MAJOR,
            suite_version=SUITE_VERSION,
            corpus_hash=corpus.hash,
            probe_layers=("generic",),
            target_type=discovery.target_type,
            similarity_backend="lexical",
            # §11.9: `judge{present, model, prompt_version}` is a HARD key,
            # so a judged run refuses to compare against an unjudged one
            # rather than mixing a model's verdicts with a regex's. It was
            # hardcoded None, which would have let exactly that through.
            judge=JudgeKey(model=judge.model, prompt_version=judge.prompt_version)
            if judge
            else None,
            profile=profile,
            pricing_source="none",
            scorer_versions=scorers,
            extraction_path=(text_path or discovery.extraction.path) or "",
        ),
        soft=SoftKeys(
            n_runs=runs,
            concurrency=DEFAULT_CONCURRENCY,
            tool_version=TOOL_VERSION,
        ),
        local=False,
    )


def _collect_skips(result: SweepResult) -> None:
    """I5: every scorer that could not run names the detector that stopped it."""
    from sweepeval.scorers.deferred import DEFERRED_REASON, DeferredScorer

    for scorer, reason in scorer_registry().skipped(result.capabilities):
        result.skipped.append((scorer.family, reason))
    for scorer in scorer_registry().all():
        if isinstance(scorer, DeferredScorer) and not any(
            f == scorer.family for f, _ in result.skipped
        ):
            result.skipped.append((scorer.family, f"{DEFERRED_REASON}: {scorer.brief}"))

    # Same reporting as the evaluate path: an entry point that raises on
    # import is indistinguishable from one that was never installed, and the
    # user is the only person who can fix either.
    from sweepeval.schema.objective import REGISTRY as _OBJECTIVES

    for name, error in scorer_registry().plugin_errors:
        result.skipped.append(
            (f"plugin:{name}", f"scorer plugin failed to load: {error}")
        )
    for name, error in _OBJECTIVES.plugin_errors:
        result.skipped.append(
            (f"plugin:{name}", f"objective plugin failed to load: {error}")
        )


def _collect_assumptions(result: SweepResult) -> None:
    """Low-confidence inferences the user can correct (§8.6)."""
    if result.discovery is None:
        return
    extraction = result.discovery.extraction
    if extraction.confidence in {"low", "medium", "none"}:
        result.assumptions.append(
            (
                "extraction.text_path",
                extraction.confidence,
                f"{extraction.path} (via {extraction.method}) — correct it in "
                "sweepeval.yaml and re-run",
            )
        )
    if result.discovery.target_type_confidence in {"low", "medium"}:
        result.assumptions.append(
            (
                "target.target_type",
                result.discovery.target_type_confidence,
                f"{result.discovery.target_type} — it gates which results may be "
                "compared (I6)",
            )
        )


def rejected_axis_lines(plan: SweepPlan) -> Sequence[str]:
    """D15's banner: every candidate axis and why it is not being swept."""
    lines = [f"{axis}: {reason}" for axis, reason in plan.rejected_axes]
    lines.extend(f"model {mid}: {reason}" for mid, reason in plan.dropped_models)
    return lines


def capability_axes(capabilities: CapabilityReport) -> dict[str, bool]:
    return {"system_prompt": capabilities.supports(Capability.SYSTEM_PROMPT)}


__all__ += ["capability_axes", "rejected_axis_lines"]
