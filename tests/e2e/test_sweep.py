"""End-to-end sweep (spec §12, D15, D22, I4, I9)."""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console
from tests.conftest import make_app, make_client

from sweepeval.execute.artifacts import read_json
from sweepeval.execute.budget import BudgetCap
from sweepeval.execute.sweep import SweepStatus, asweep_target
from sweepeval.report.sweep import render_sweep


async def _sweep(scenario: str, tmp_path: Path, **kw):
    app = make_app(scenario)
    client = make_client(app)
    try:
        return await asweep_target(
            "https://mock.test" + app.scenario.paths[0],
            key=kw.pop("key", "test-key-abcdefgh"),
            client=client,
            root=str(tmp_path),
            runs=kw.pop("runs", 2),
            profile=kw.pop("profile", "quick"),
            authorized=True,
            authorization_prompt=False,
            seed=7,
            config_cap=kw.pop("config_cap", 2),
            **kw,
        )
    finally:
        await client.aclose()


@pytest.fixture(scope="module")
def swept(tmp_path_factory):
    """One real sweep, shared. Every assertion below reads the same run."""
    import asyncio

    return asyncio.run(
        _sweep("openai_clean", tmp_path_factory.mktemp("sweep"))
    )


def test_a_sweep_executes_every_planned_configuration(swept) -> None:
    assert swept.status is SweepStatus.COMPLETE
    assert len(swept.configs) == len(swept.plan.configs) >= 2
    assert not swept.not_run


def test_every_configuration_produces_metrics(swept) -> None:
    for row in swept.configs:
        assert row.metrics, row.config_id
        assert "security_pass_rate" in row.metrics


def test_the_probe_set_is_identical_across_configurations(swept) -> None:
    """I4, checked from the run rather than from the code.

    A paired comparison pairs on the probe. Two configs scored on different
    probes cannot be compared that way, and every interval downstream assumes
    the pairing holds.
    """
    per_config = {
        row.config_id: sorted({o.unit.unit_id for o in row.outcomes if o.run_idx >= 0})
        for row in swept.configs
    }
    reference = next(iter(per_config.values()))
    for config_id, units in per_config.items():
        assert units == reference, config_id


def test_the_canary_table_is_shared_and_written_once(swept) -> None:
    """I4 from the artifact (§8.1)."""
    plan = read_json(swept.store.plan_path)
    assert plan["canary_table"]
    assert plan["units"]
    for config in plan["configs"]:
        assert "canary_table" not in config


def test_the_plan_and_manifest_are_written_before_scoring(swept) -> None:
    assert swept.store.plan_path.exists()
    manifest = read_json(swept.store.manifest_path)
    assert manifest["plan_hash"]
    assert manifest["comparability"]["hard"]["corpus_hash"] == swept.corpus.hash
    assert manifest["heuristics"]["tokens"] == "chars/4"


def test_the_manifest_carries_no_credential(swept) -> None:
    text = swept.store.manifest_path.read_text(encoding="utf-8")
    assert "test-key-abcdefgh" not in text


def test_the_sweep_reports_which_axes_it_rejected(swept) -> None:
    """D15: name every candidate axis and why it is not being swept."""
    assert swept.plan.axes
    assert swept.plan.rejected_axes or swept.plan.shrink_steps


def test_the_sampling_test_actually_ran(swept) -> None:
    """§9.1. Without it every axis reads 'sampling-effect test not run'."""
    assert swept.sampling.verdicts, swept.sampling.not_tested
    assert "temperature" in swept.sampling.verdicts


def test_the_report_renders(swept) -> None:
    console = Console(record=True, width=200)
    render_sweep(swept, console)
    text = console.export_text()
    assert "sweep" in text
    assert "COMPLETE" in text


# --- the pre-flight gate (I9) ---------------------------------------------


async def test_a_declined_estimate_sends_nothing(tmp_path: Path) -> None:
    """I9: no billable request of any kind precedes the confirmation."""
    seen: list[object] = []

    def refuse(estimate):
        seen.append(estimate)
        return False

    result = await _sweep("openai_clean", tmp_path, confirm=refuse)
    assert result.status is SweepStatus.DECLINED
    assert seen, "the estimate was never shown"
    assert result.store is None
    assert not (tmp_path / "runs").exists()
    assert result.discovery is None


async def test_the_estimate_covers_discovery_not_just_scoring(tmp_path: Path) -> None:
    captured: list[object] = []

    def refuse(estimate):
        captured.append(estimate)
        return False

    await _sweep("openai_clean", tmp_path, confirm=refuse)
    phases = {p.phase for p in captured[0].phases}  # type: ignore[attr-defined]
    assert "discovery" in phases
    assert "capabilities" in phases


async def test_a_cap_that_cannot_afford_discovery_sends_nothing(
    tmp_path: Path,
) -> None:
    """Distinct from a cap below the estimate, which buys a partial sweep.

    A cap below discovery and capability detection buys no sweep at all, and
    spending it on discovery would leave nothing to score with.
    """
    result = await _sweep(
        "openai_clean", tmp_path, cap=BudgetCap(value=10, unit="requests")
    )
    assert result.status is SweepStatus.DECLINED
    assert "no configuration could be scored" in result.stop_reason
    assert result.discovery is None


# --- the budget cap during execution (§12.3, §12.5) ------------------------


async def test_the_cap_stops_between_configs_and_names_what_never_ran(
    tmp_path: Path,
) -> None:
    """A frontier over a subset presented as a frontier over the sweep is a
    lie the artifact cannot detect later."""
    result = await _sweep(
        "openai_clean",
        tmp_path,
        config_cap=3,
        cap=BudgetCap(value=100_000, unit="requests"),
    )
    # The cap above is generous; re-run with one that binds after config 1.
    binding = await _sweep(
        "openai_clean",
        tmp_path / "b",
        config_cap=3,
        cap=BudgetCap(value=result.configs[0].requests + 50, unit="requests"),
    )
    assert binding.status is SweepStatus.INCOMPLETE
    assert binding.not_run
    assert len(binding.configs) < len(binding.plan.configs)
    assert "stopped at the requests cap" in binding.stop_reason

    console = Console(record=True, width=200)
    render_sweep(binding, console)
    text = console.export_text()
    assert "INCOMPLETE" in text
    assert "never ran" in text


# --- resume (§12.5) --------------------------------------------------------


async def test_resuming_an_unchanged_run_refuses_nothing(tmp_path: Path) -> None:
    first = await _sweep("openai_clean", tmp_path, config_cap=2)
    again = await _sweep(
        "openai_clean", tmp_path, config_cap=2, resume_run_id=first.run_id
    )
    assert again.status is SweepStatus.COMPLETE
    assert again.resume is not None and again.resume.ok
    # Every unit-run was already checkpointed, so the resume sends nothing new.
    assert sum(row.requests for row in again.configs) == 0


async def test_resuming_with_a_changed_sweep_refuses(tmp_path: Path) -> None:
    """A hand-edited config between runs refuses rather than mixing."""
    first = await _sweep("openai_clean", tmp_path, config_cap=2)
    changed = await _sweep(
        "openai_clean", tmp_path, config_cap=3, resume_run_id=first.run_id
    )
    assert changed.status is SweepStatus.REFUSED
    assert changed.resume is not None
    assert any(r.key == "plan_hash" for r in changed.resume.refusals)
    assert "would mix rows" in changed.stop_reason


# --- the empty sweep (D15) -------------------------------------------------


async def test_a_target_with_no_axes_is_an_evaluation_with_a_banner(
    tmp_path: Path,
) -> None:
    """The modal outcome for a custom agent system, not an edge case."""
    # A scenario with genuinely nothing to sweep: no model list, a system role
    # it drops, an inert temperature. `weird_shape` used to serve here only
    # because its capability probes were malformed and every capability came
    # back UNSUPPORTED -- the test was passing for the wrong reason.
    result = await _sweep("no_axes", tmp_path)
    assert result.single_config
    assert len(result.configs) == 1
    assert result.plan.rejected_axes

    console = Console(record=True, width=200)
    render_sweep(result, console)
    text = console.export_text()
    assert "single-configuration evaluation" in text
    assert "rejected axes" in text


# --- cache detection (§12.7, D41) ------------------------------------------


async def test_a_caching_target_is_flagged_and_the_flag_reaches_the_metrics(
    tmp_path: Path,
) -> None:
    result = await _sweep("caches_responses", tmp_path, config_cap=1)
    row = result.configs[0]
    assert row.cache.suspected, row.cache.reason

    from sweepeval.schema.metric import Flag

    determinism = row.metrics.get("target_determinism_at_temp0")
    assert determinism is not None
    assert Flag.CACHE_SUSPECTED in determinism.flags
    assert Flag.CACHE_SUSPECTED not in row.metrics["security_pass_rate"].flags

    console = Console(record=True, width=200)
    render_sweep(result, console)
    assert "CACHE_SUSPECTED" in console.export_text()


async def test_a_non_caching_target_is_not_flagged(tmp_path: Path) -> None:
    """The detector has to discriminate, not just fire."""
    result = await _sweep("openai_clean", tmp_path, config_cap=1)
    assert not result.configs[0].cache.suspected, result.configs[0].cache.reason


# --- ranking, end to end (§14) ---------------------------------------------


def test_the_frontier_is_computed_over_a_real_sweep(swept) -> None:
    from sweepeval.pipeline import rank_sweep

    frontier = rank_sweep(swept, seed=7)
    assert frontier.frontier, "no configuration reached the frontier"
    assert frontier.clusters
    for cluster in frontier.clusters:
        for member in cluster.members:
            assert member in swept.config_ids


def test_determinism_is_shared_across_temperature_siblings(swept) -> None:
    """§14.2, from a real run: measured per row, every temp=0 config would be
    non-dominated for free on a metric restating its own label."""
    by_temp: dict[float, list[str]] = {}
    for row in swept.configs:
        by_temp.setdefault(row.config.params.get("temperature", 0.0), []).append(
            row.config_id
        )
    if len(by_temp) < 2:
        import pytest as _pytest

        _pytest.skip("this sweep has no temperature axis to share across")

    values = {
        row.config_id: row.metrics["target_determinism_at_temp0"].point
        for row in swept.configs
    }
    assert len(set(values.values())) < len(values), values
    assert swept.determinism_scope


def test_the_frontier_document_is_written(tmp_path: Path) -> None:
    from sweepeval.pipeline import rank_sweep
    from sweepeval.report.frontier_json import frontier_payload

    result = await_sweep(tmp_path)
    frontier = rank_sweep(result, seed=7)
    payload = frontier_payload(frontier)
    assert payload["comparisons"]
    assert payload["objectives"]
    assert "rank" not in payload


def await_sweep(tmp_path: Path):
    import asyncio

    return asyncio.run(_sweep("openai_clean", tmp_path, config_cap=2))


async def test_narrowing_objectives_needs_no_rerun(tmp_path: Path) -> None:
    """§14.1: every config runs the full profile, so narrowing offline is safe."""
    from sweepeval.pipeline import rank_sweep

    result = await _sweep("openai_clean", tmp_path, config_cap=2)
    wide = rank_sweep(result, seed=7)
    narrow = rank_sweep(result, objectives=["security", "latency"], seed=7)
    assert len(narrow.objectives) == 2
    assert len(wide.objectives) == 6


async def test_an_unknown_objective_names_what_is_available(tmp_path: Path) -> None:
    from sweepeval.pipeline import rank_sweep

    result = await _sweep("openai_clean", tmp_path, config_cap=1)
    with pytest.raises(ValueError, match="Available"):
        rank_sweep(result, objectives=["not_a_real_objective"])
