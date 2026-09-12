"""Single-configuration evaluation (spec §12.1, §4.2).

The whole pipeline for one config: discover, detect capabilities, plan the
frozen probe set, execute N runs, aggregate.

This is also the empty-sweep path (D15). When no axis survives discovery, a
sweep *is* a single-configuration evaluation, and the report says so rather
than presenting a frontier of one as though a search had happened. For a
custom agent system — the target type the black-box positioning is built for —
that is the modal outcome, not an edge case.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from sweepeval.capabilities.detect import (
    Capability,
    CapabilityBudget,
    CapabilityReport,
    detect_all,
)
from sweepeval.corpus.loader import Corpus, load_corpus
from sweepeval.corpus.template import Profile
from sweepeval.discovery.budget import DiscoveryBudget
from sweepeval.discovery.runner import DiscoveryOutcome, discover_target
from sweepeval.execute.authz import AuthorizationRecord, AuthorizationStore, require_authorization
from sweepeval.execute.runner import RunPlan, UnitOutcome, execute_config
from sweepeval.http.client import TransportClient
from sweepeval.http.governor import Governor
from sweepeval.schema.comparability import Comparability, HardKeys, SoftKeys
from sweepeval.schema.metric import Estimand, MetricValue
from sweepeval.schema.objective import REGISTRY
from sweepeval.schema.observation import Observation
from sweepeval.schema.unit import Unit
from sweepeval.schema.versions import SCHEMA_MAJOR, SUITE_VERSION, TOOL_VERSION
from sweepeval.scorers import registry as scorer_registry
from sweepeval.stats.aggregate import ClusterTable, aggregate_metric, build_cluster_table
from sweepeval.store.redaction import Redactor
from sweepeval.store.run import Store, new_run_id

__all__ = ["EvaluationResult", "aevaluate_target"]

_RATE_METRICS = (
    "security_pass_rate",
    "guardrail_pass_rate",
    "config_repeatability",
    "target_determinism_at_temp0",
    "semantic_stability",
    "invariance",
)


@dataclass
class EvaluationResult:
    run_id: str
    config_id: str
    profile: Profile
    corpus: Corpus
    discovery: DiscoveryOutcome
    capabilities: CapabilityReport
    comparability: Comparability
    authorization: AuthorizationRecord | None
    outcomes: list[UnitOutcome] = field(default_factory=list)
    metrics: dict[str, MetricValue] = field(default_factory=dict)
    clusters: dict[str, dict[str, float]] = field(default_factory=dict)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    assumptions: list[tuple[str, str, str]] = field(default_factory=list)
    axes_rejected: list[tuple[str, str]] = field(default_factory=list)
    families_not_run: tuple[str, ...] = ()
    """Families whose probes were not executed because a capability ruled them
    out. Their calls are not spent and their metrics do not appear."""

    retention_curve: dict[int, float] = field(default_factory=dict)
    retention_weights: dict[int, float] = field(default_factory=dict)
    retention_depth_at_floor: int | None = None
    store: Store | None = None

    @property
    def observations(self) -> list[Observation]:
        return [o for outcome in self.outcomes for o in outcome.observations]

    @property
    def single_config(self) -> bool:
        return True


async def aevaluate_target(
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
) -> EvaluationResult:
    """Discover, detect, plan, execute and aggregate one configuration."""
    from pathlib import Path

    owned = client is None
    http = client or httpx.AsyncClient(timeout=60.0, follow_redirects=True)

    try:
        discovery = await discover_target(
            http, url, key, budget=DiscoveryBudget(), seed=seed
        )
        capabilities = await detect_all(
            http, discovery.ladder, key, discovery.extraction.path,
            budget=CapabilityBudget(), profile=profile,
        )

        corpus = load_corpus(profile)

        authorization = require_authorization(
            url,
            store=AuthorizationStore(Path(root)),
            flag=authorized,
            prompt=authorization_prompt,
        )

        run_id = new_run_id()
        redactor = Redactor(secrets=[key] if key else [])
        store = Store(Path(root), run_id, redactor, store_text=store_text)

        # Do not execute units for a family the capability report has already
        # ruled out. Running them would spend real money producing rows for a
        # family the report says is SKIPPED — a direct contradiction, and the
        # context family alone is over half the calls at `standard`.
        skipped_families = {s.family for s, _ in scorer_registry().skipped(capabilities)}
        units = tuple(
            t.to_unit() for t in corpus.probes if t.family not in skipped_families
        )
        not_run = sorted({t.family for t in corpus.probes} & skipped_families)
        plan = RunPlan(
            config_id="default",
            units=units,
            runs=runs,
            master_seed=f"{run_id}:{seed}",
        )

        transport = TransportClient(
            http, Governor(seed=seed), run_id=run_id, shape=discovery.ladder.shape.name
        )
        outcomes = await execute_config(
            transport, plan, discovery.ladder,
            text_path=discovery.extraction.path,
            store=store, registry=scorer_registry(),
            capabilities=capabilities, key=key,
        )
    finally:
        if owned:
            await http.aclose()

    result = EvaluationResult(
        run_id=run_id,
        config_id="default",
        profile=profile,
        corpus=corpus,
        discovery=discovery,
        capabilities=capabilities,
        comparability=_comparability(discovery, corpus, profile, runs),
        authorization=authorization,
        outcomes=outcomes,
        store=store,
    )
    result.families_not_run = tuple(not_run)
    _aggregate(result, seed=seed)
    _collect_skips(result)
    _collect_assumptions(result)
    _collect_rejected_axes(result)
    return result


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


def _aggregate(result: EvaluationResult, *, seed: int) -> None:
    observations = result.observations
    indicative = result.profile == "quick"

    for metric in _RATE_METRICS:
        table = build_cluster_table(
            observations, metric=metric, config_id=result.config_id
        )
        result.clusters[metric] = dict(table.values)
        # Emitted even with no scored cluster. I5: a metric that vanishes from
        # the report is indistinguishable from one that passed, and "every
        # trial was unscorable" is a finding, not an absence.
        result.metrics[metric] = aggregate_metric(
            table, estimand=Estimand.generalization, seed=seed, indicative=indicative
        )

    _aggregate_context(result, seed=seed, indicative=indicative)
    _aggregate_operational(result, seed=seed, indicative=indicative)


def _aggregate_context(
    result: EvaluationResult, *, seed: int, indicative: bool
) -> None:
    """Recall per conversation, plus the depth-weighted AUC (§11.5)."""
    from sweepeval.scorers.context import depth_at_floor, retention_auc, retention_curve

    observations = result.observations
    table = build_cluster_table(
        observations, metric="fact_recall", config_id=result.config_id,
        stratify_by_depth=True,
    )
    if not table.coverage.attempted:
        return

    result.clusters["fact_recall"] = dict(table.values)
    # Stratified: an unstratified resample can empty a depth, and the trapezoid
    # is undefined there (§13.3).
    result.metrics["fact_recall"] = aggregate_metric(
        table, seed=seed, indicative=indicative
    )

    curve = retention_curve(observations)
    auc, weights = retention_auc(curve)
    result.retention_curve = curve
    result.retention_weights = weights
    result.retention_depth_at_floor = depth_at_floor(curve)

    # The AUC resamples conversations, so it reuses fact_recall's clusters but
    # reports the depth-weighted quantity.
    result.clusters["context_retention_auc"] = dict(table.values)
    result.metrics["context_retention_auc"] = aggregate_metric(
        table, seed=seed, indicative=indicative
    ).model_copy(update={"point": auc})


def _aggregate_operational(
    result: EvaluationResult, *, seed: int, indicative: bool
) -> None:
    observations = result.observations
    for metric, bounded in (("latency_ms", False), ("tokens_out", False), ("error_rate", True)):
        table = build_cluster_table(
            observations, metric=metric, config_id=result.config_id
        )
        if not table.coverage.attempted:
            continue
        result.clusters[metric] = dict(table.values)
        result.metrics[metric] = aggregate_metric(
            table, seed=seed, bounded=bounded, indicative=indicative
        )


def _collect_skips(result: EvaluationResult) -> None:
    """I5: every scorer that could not run names the detector that stopped it."""
    for scorer, reason in scorer_registry().skipped(result.capabilities):
        result.skipped.append((scorer.family, reason))

    from sweepeval.scorers.deferred import DEFERRED_REASON, DeferredScorer

    for scorer in scorer_registry().all():
        if isinstance(scorer, DeferredScorer) and not any(
            f == scorer.family for f, _ in result.skipped
        ):
            result.skipped.append((scorer.family, f"{DEFERRED_REASON}: {scorer.brief}"))


def _collect_assumptions(result: EvaluationResult) -> None:
    """Low-confidence inferences the user can correct (§8.6)."""
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
                f"{result.discovery.target_type} — it gates which results may "
                "be compared (I6)",
            )
        )


def _collect_rejected_axes(result: EvaluationResult) -> None:
    """D15: name every candidate axis and why it is not being swept."""
    caps = result.capabilities
    models = result.discovery.ladder.sniff.models

    if not caps.supports(Capability.SYSTEM_PROMPT):
        result.axes_rejected.append(
            ("system_prompt", caps.skip_reason(Capability.SYSTEM_PROMPT))
        )
    if len(models) <= 1:
        result.axes_rejected.append(
            ("model", f"{len(models)} model identifier(s) discovered")
        )
    result.axes_rejected.append(
        ("temperature", "sampling-effect test not run in single-config evaluation")
    )


def default_objectives() -> Sequence[Any]:
    return REGISTRY.defaults()


def unit_ids(units: Sequence[Unit]) -> tuple[str, ...]:
    return tuple(u.unit_id for u in units)


def cluster_view(table: ClusterTable) -> dict[str, float]:
    return dict(table.values)


__all__ += ["cluster_view", "default_objectives", "unit_ids"]
