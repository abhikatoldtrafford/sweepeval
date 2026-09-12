"""``aggregates.json`` and offline reporting (spec §6.2, §15, D33).

A report has to be reproducible from the stored run alone: no network, no
credentials, no re-run. That is what lets a committed example run be the
README's evidence (D33), what lets CI re-render a report from an artifact, and
what makes "changing your ``--prefer`` needs no re-run" true rather than
aspirational.

So the sweep writes ``aggregates.json`` — every config's intervals and the
cluster tables behind them — and this module reads it back into objects shaped
like the ones the renderers already take. The renderers are duck-typed on
purpose: one set of them serves a live run and a stored one, which is the only
way the two stay identical.

``clusters`` are stored, not just the summaries. They are what the paired test
resamples, so without them a stored run could be re-reported but never
re-ranked, and ``--prefer`` and ``--objectives`` would silently require the
network after all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sweepeval.schema.metric import MetricValue
from sweepeval.store.json_io import read_json, write_json

__all__ = [
    "StoredConfig",
    "StoredRun",
    "aggregates_payload",
    "load_run",
    "write_aggregates",
]


@dataclass
class _Spec:
    """The subset of ``ConfigSpec`` the renderers touch."""

    config_id: str
    params: dict[str, Any]
    system_prompt_variant: str
    _label: str

    def label(self) -> str:
        return self._label


@dataclass
class _Cache:
    suspected: bool = False
    reason: str = ""
    evidence: tuple[Any, ...] = ()


@dataclass
class _Cost:
    text: str = ""

    def describe(self) -> str:
        return self.text


@dataclass
class _HardFails:
    confirmed: tuple[Any, ...] = ()
    count: int = 0


@dataclass
class StoredConfig:
    config: _Spec
    metrics: dict[str, MetricValue] = field(default_factory=dict)
    clusters: dict[str, dict[str, float]] = field(default_factory=dict)
    strata: dict[str, dict[str, str]] = field(default_factory=dict)
    cache: _Cache = field(default_factory=_Cache)
    cost: _Cost | None = None
    hard_fails: _HardFails = field(default_factory=_HardFails)
    coverage: dict[str, tuple[int, int]] = field(default_factory=dict)
    requests: int = 0

    @property
    def config_id(self) -> str:
        return self.config.config_id


@dataclass
class _Status:
    value: str


@dataclass
class _Corpus:
    unit_count: int
    calls_per_run: int
    hash: str = ""


@dataclass
class _Plan:
    configs: tuple[_Spec, ...] = ()
    axes: dict[str, list[Any]] = field(default_factory=dict)
    shrink_steps: list[str] = field(default_factory=list)
    rejected_axes: list[tuple[str, str]] = field(default_factory=list)
    dropped_models: list[tuple[str, str]] = field(default_factory=list)
    cap: int = 12


@dataclass
class _Discovery:
    payload: dict[str, Any]


@dataclass
class StoredRun:
    """A sweep read back from disk, shaped like the live one."""

    run_id: str
    profile: str
    runs: int
    status: _Status
    corpus: _Corpus
    plan: _Plan
    configs: list[StoredConfig] = field(default_factory=list)
    not_run: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    assumptions: list[tuple[str, str, str]] = field(default_factory=list)
    families_not_run: tuple[str, ...] = ()
    determinism_scope: dict[str, str] = field(default_factory=dict)
    stop_reason: str = ""
    discovery: _Discovery | None = None
    store: Any = None
    run_dir: Path | None = None

    @property
    def single_config(self) -> bool:
        return len(self.plan.configs) <= 1

    @property
    def config_ids(self) -> tuple[str, ...]:
        return tuple(c.config_id for c in self.configs)


def aggregates_payload(sweep: Any) -> dict[str, Any]:
    """Everything needed to re-report and re-rank a run offline."""
    return {
        "run_id": sweep.run_id,
        "profile": sweep.profile,
        "runs": sweep.runs,
        "status": sweep.status.value,
        "stop_reason": sweep.stop_reason,
        "corpus": {
            "units": sweep.corpus.unit_count,
            "calls_per_run": sweep.corpus.calls_per_run,
            "hash": sweep.corpus.hash,
        },
        "target": (
            dict(sweep.discovery.payload["target"]) if sweep.discovery else {}
        ),
        "plan": {
            "cap": sweep.plan.cap,
            "axes": {k: list(v) for k, v in sorted(sweep.plan.axes.items())},
            "shrink_steps": list(sweep.plan.shrink_steps),
            "rejected_axes": [[a, r] for a, r in sweep.plan.rejected_axes],
            "dropped_models": [[m, r] for m, r in sweep.plan.dropped_models],
            "configs": [
                {
                    "config_id": c.config_id,
                    "label": c.label(),
                    "params": dict(sorted(c.params.items())),
                    "system_prompt_variant": c.system_prompt_variant,
                }
                for c in sweep.plan.configs
            ],
        },
        "configs": [_config_payload(row) for row in sweep.configs],
        "not_run": list(sweep.not_run),
        "skipped": [[f, r] for f, r in sorted(sweep.skipped)],
        "assumptions": [list(a) for a in sweep.assumptions],
        "families_not_run": list(sweep.families_not_run),
        "determinism_scope": dict(sorted(sweep.determinism_scope.items())),
    }


def _config_payload(row: Any) -> dict[str, Any]:
    return {
        "config_id": row.config_id,
        "label": row.config.label(),
        "params": dict(sorted(row.config.params.items())),
        "system_prompt_variant": row.config.system_prompt_variant,
        "requests": row.requests,
        "metrics": {
            name: value.model_dump(mode="json")
            for name, value in sorted(row.metrics.items())
        },
        "clusters": {
            metric: dict(sorted(values.items()))
            for metric, values in sorted(row.clusters.items())
        },
        "strata": {
            metric: dict(sorted(values.items()))
            for metric, values in sorted(row.strata.items())
        },
        "cache": {"suspected": row.cache.suspected, "reason": row.cache.reason},
        "cost": row.cost.describe() if row.cost is not None else None,
        "hard_fails": [h.describe() for h in row.hard_fails.confirmed],
        "coverage": {
            family: list(pair) for family, pair in sorted(row.coverage.items())
        },
    }


def write_aggregates(sweep: Any) -> Path | None:
    """Write ``aggregates.json`` beside the logs it was derived from."""
    if sweep.store is None:
        return None
    path = sweep.store.aggregates_path
    write_json(path, aggregates_payload(sweep))
    return path


def load_run(run_dir: Path | str) -> StoredRun:
    """Read a stored run back. Sends nothing and needs no credentials."""
    directory = Path(run_dir)
    payload = read_json(directory / "aggregates.json")
    if not payload:
        raise FileNotFoundError(
            f"no aggregates.json in {directory} — a run can only be re-reported "
            "if it was stored; re-run with a --root you can read back"
        )

    plan_payload = payload.get("plan", {})
    specs = tuple(
        _Spec(
            config_id=c["config_id"],
            params=dict(c.get("params", {})),
            system_prompt_variant=c.get("system_prompt_variant", "none"),
            _label=c.get("label", c["config_id"]),
        )
        for c in plan_payload.get("configs", [])
    )

    corpus = payload.get("corpus", {})
    run = StoredRun(
        run_id=str(payload.get("run_id", "")),
        profile=str(payload.get("profile", "quick")),
        runs=int(payload.get("runs", 0)),
        status=_Status(str(payload.get("status", "COMPLETE"))),
        corpus=_Corpus(
            unit_count=int(corpus.get("units", 0)),
            calls_per_run=int(corpus.get("calls_per_run", 0)),
            hash=str(corpus.get("hash", "")),
        ),
        plan=_Plan(
            configs=specs,
            axes=dict(plan_payload.get("axes", {})),
            shrink_steps=list(plan_payload.get("shrink_steps", [])),
            rejected_axes=[tuple(r) for r in plan_payload.get("rejected_axes", [])],
            dropped_models=[tuple(d) for d in plan_payload.get("dropped_models", [])],
            cap=int(plan_payload.get("cap", 12)),
        ),
        not_run=list(payload.get("not_run", [])),
        skipped=[tuple(s) for s in payload.get("skipped", [])],
        assumptions=[tuple(a) for a in payload.get("assumptions", [])],
        families_not_run=tuple(payload.get("families_not_run", [])),
        determinism_scope=dict(payload.get("determinism_scope", {})),
        stop_reason=str(payload.get("stop_reason", "")),
        discovery=_Discovery({"target": payload.get("target", {})}),
        run_dir=directory,
    )

    for row in payload.get("configs", []):
        run.configs.append(
            StoredConfig(
                config=_Spec(
                    config_id=row["config_id"],
                    params=dict(row.get("params", {})),
                    system_prompt_variant=row.get("system_prompt_variant", "none"),
                    _label=row.get("label", row["config_id"]),
                ),
                metrics={
                    name: MetricValue.model_validate(value)
                    for name, value in row.get("metrics", {}).items()
                },
                clusters={
                    metric: {k: float(v) for k, v in values.items()}
                    for metric, values in row.get("clusters", {}).items()
                },
                strata={
                    metric: dict(values)
                    for metric, values in row.get("strata", {}).items()
                },
                cache=_Cache(
                    suspected=bool(row.get("cache", {}).get("suspected", False)),
                    reason=str(row.get("cache", {}).get("reason", "")),
                ),
                cost=_Cost(row["cost"]) if row.get("cost") else None,
                hard_fails=_HardFails(count=len(row.get("hard_fails", []))),
                coverage={
                    family: (int(pair[0]), int(pair[1]))
                    for family, pair in row.get("coverage", {}).items()
                },
                requests=int(row.get("requests", 0)),
            )
        )
    return run
