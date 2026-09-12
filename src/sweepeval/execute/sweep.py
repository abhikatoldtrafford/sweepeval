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

from collections.abc import Callable, Sequence
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
from sweepeval.execute.planner import CAP_BY_PROFILE, ConfigSpec, SweepPlan, plan_sweep
from sweepeval.execute.runner import RunPlan, UnitOutcome, execute_config
from sweepeval.http.client import TransportClient
from sweepeval.http.governor import Governor
from sweepeval.schema.call import Call
from sweepeval.schema.comparability import Comparability, HardKeys, SoftKeys
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
    planned_cap = min(config_cap or CAP_BY_PROFILE[profile], CAP_BY_PROFILE[profile])
    estimate = estimate_run(
        corpus,
        configs=planned_cap,
        runs=runs,
        profile=profile,
        pricing_source=pricing.source if pricing else "none",
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
    if budget_cap.forbids_starting(estimate):
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
        )
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
        comparability = _comparability(discovery, corpus, profile, runs)

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
                    "extraction_path": discovery.extraction.path or "",
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

        spent = _already_spent(store, plan)
        for index, config in enumerate(plan.configs):
            # The cap is checked between configs, never mid-config: stopping
            # inside one leaves a config with partial coverage, and a partially
            # covered config is worse than an absent one — it can be compared
            # against, and the comparison is wrong (§14.5).
            if _cap_reached(budget_cap, spent, estimate):
                result.status = SweepStatus.INCOMPLETE
                result.not_run = [c.config_id for c in plan.configs[index:]]
                result.stop_reason = (
                    f"stopped at the {budget_cap.unit} cap of {budget_cap.value} "
                    f"after {index} of {len(plan.configs)} configs"
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
                text_path=discovery.extraction.path,
                store=store,
                key=key,
                seed=seed,
                profile=profile,
                pricing=pricing,
            )
            spent += row.requests
            result.configs.append(row)
    finally:
        if owned:
            await http.aclose()

    _collect_skips(result)
    _collect_assumptions(result)
    return result


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
) -> ConfigResult:
    """Execute one configuration and aggregate it."""
    run_plan = RunPlan(
        config_id=config.config_id,
        units=units,
        runs=runs,
        master_seed=master_seed,
        params=dict(config.params),
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
    row.requests = len(row.calls)
    row.cache = detect_cache(row.calls, config_id=config.config_id)
    row.cost = account(row.calls, pricing=pricing)

    aggregate = aggregate_config(
        row.observations,
        config_id=config.config_id,
        seed=seed,
        indicative=profile == "quick",
    )
    # §12.7: the flag rides on every metric a cache would corrupt, attached
    # here rather than in the aggregator, which has no view of the calls.
    aggregate.metrics = flag_affected(aggregate.metrics, row.cache)
    row.aggregate = aggregate
    return row


def _cap_reached(cap: BudgetCap, spent: int, estimate: Estimate) -> bool:
    if cap.value is None:
        return False
    if cap.unit == "requests":
        return spent >= cap.value
    if cap.unit == "tokens":
        # Tokens are not counted per config as they arrive, so the cap binds
        # on the estimate's own token-per-request ratio rather than pretending
        # to a precision the counter does not have.
        per_request = estimate.total_tokens / max(1, estimate.total_requests)
        return spent * per_request >= cap.value
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


def _comparability(
    discovery: DiscoveryOutcome, corpus: Corpus, profile: Profile, runs: int
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
            judge=None,
            profile=profile,
            pricing_source="none",
            scorer_versions=scorers,
            extraction_path=discovery.extraction.path or "",
        ),
        soft=SoftKeys(n_runs=runs, concurrency=2, tool_version=TOOL_VERSION),
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
