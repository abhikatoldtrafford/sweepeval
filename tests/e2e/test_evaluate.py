"""End-to-end single-configuration evaluation (spec §12.1, §15)."""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console
from tests.conftest import make_app, make_client

from sweepeval.execute.evaluate import aevaluate_target
from sweepeval.report.terminal import render_evaluation
from sweepeval.schema.metric import Flag


async def _evaluate(scenario: str, tmp_path: Path, key: str | None = None, **kw):
    app = make_app(scenario)
    client = make_client(app)
    try:
        return await aevaluate_target(
            "https://mock.test" + app.scenario.paths[0],
            key=key, client=client, root=str(tmp_path), runs=2,
            authorized=True, authorization_prompt=False, seed=7, **kw,
        )
    finally:
        await client.aclose()


async def test_the_whole_pipeline_produces_metrics(tmp_path: Path) -> None:
    result = await _evaluate("openai_clean", tmp_path, key="test-key-abcdefgh")
    assert result.metrics, "an evaluation with no metric is not an evaluation"
    assert "security_pass_rate" in result.metrics
    assert "guardrail_pass_rate" in result.metrics


async def test_every_metric_carries_an_interval_or_declines_one(tmp_path: Path) -> None:
    """I3, end to end."""
    result = await _evaluate("openai_clean", tmp_path, key="test-key-abcdefgh")
    for name, value in result.metrics.items():
        has_bounds = value.lo is not None and value.hi is not None
        declined = Flag.NO_VALID_INTERVAL in value.flags
        assert has_bounds or declined, name


async def test_artifacts_are_written(tmp_path: Path) -> None:
    result = await _evaluate("openai_clean", tmp_path, key="test-key-abcdefgh")
    run_dir = result.store.run_dir
    assert (run_dir / "calls.jsonl").exists()
    assert (run_dir / "observations.jsonl").exists()
    assert result.store.calls.count() > 0
    assert result.store.observations.count() > 0


async def test_a_leaky_target_scores_worse_on_security(tmp_path: Path) -> None:
    """The measurement has to actually discriminate."""
    clean = await _evaluate("openai_clean", tmp_path / "a", key="test-key-abcdefgh")
    leaky = await _evaluate("leaky_guardrails", tmp_path / "b")
    assert clean.metrics["security_pass_rate"].point >= (
        leaky.metrics["security_pass_rate"].point
    )


async def test_a_refusing_target_passes_security_and_guardrails(tmp_path: Path) -> None:
    """§11.8: a refusal is correct behaviour on both families."""
    result = await _evaluate("refuses_everything", tmp_path)
    assert result.metrics["security_pass_rate"].point == 1.0
    assert result.metrics["guardrail_pass_rate"].point == 1.0


async def test_quick_results_are_flagged_indicative(tmp_path: Path) -> None:
    result = await _evaluate("openai_clean", tmp_path, key="test-key-abcdefgh")
    assert any(Flag.INDICATIVE in v.flags for v in result.metrics.values())


async def test_deferred_families_are_reported_as_skipped(tmp_path: Path) -> None:
    """I5: a family absent from a report is indistinguishable from one that passed."""
    result = await _evaluate("openai_clean", tmp_path, key="test-key-abcdefgh")
    families = {f for f, _ in result.skipped}
    assert {"tool_integrity", "retrieval", "degradation"} <= families
    for _, reason in result.skipped:
        assert reason


async def test_rejected_axes_are_named(tmp_path: Path) -> None:
    """D15: a frontier of one says why it is a frontier of one."""
    result = await _evaluate("openai_clean", tmp_path, key="test-key-abcdefgh")
    axes = {a for a, _ in result.axes_rejected}
    assert "system_prompt" in axes
    for _, reason in result.axes_rejected:
        assert reason


async def test_comparability_keys_are_populated(tmp_path: Path) -> None:
    result = await _evaluate("openai_clean", tmp_path, key="test-key-abcdefgh")
    hard = result.comparability.hard
    assert hard.corpus_hash
    assert hard.extraction_path
    assert hard.scorer_versions
    assert hard.profile == "quick"


async def test_the_terminal_report_shows_intervals_skips_and_assumptions(
    tmp_path: Path,
) -> None:
    result = await _evaluate("weird_shape", tmp_path)
    console = Console(record=True, width=120)
    render_evaluation(result, console)
    text = console.export_text()

    assert "metrics" in text
    assert "SKIPPED" in text
    assert "coverage" in text
    assert "assumptions you can correct" in text, "target_type is low confidence here"


async def test_resume_skips_completed_unit_runs(tmp_path: Path) -> None:
    """§12.5: checkpointing is per unit-run, not per config."""
    result = await _evaluate("openai_clean", tmp_path, key="test-key-abcdefgh")
    state = result.store.state_for("default")
    completed = state.completed_unit_runs("default")
    assert len(completed) == result.corpus.unit_count * 2, "units x runs"


# --- an unmeasurable metric shows no number -------------------------------


async def test_an_unmeasurable_metric_shows_no_number(tmp_path: Path) -> None:
    """A NO_VALID_INTERVAL metric carries a placeholder point.

    Printing it reads as a measurement: "guardrail_pass_rate 0.000" says the
    target failed every guardrail, when what happened is that nothing could be
    scored. The renderers show an em dash instead.
    """
    from sweepeval.report.machine import as_json, as_markdown
    from sweepeval.schema.metric import Flag

    result = await _evaluate("openai_clean", tmp_path, key="test-key-abcdefgh")
    unmeasured = [
        name for name, v in result.metrics.items() if Flag.NO_VALID_INTERVAL in v.flags
    ]
    assert unmeasured, "the mock replies 'OK', so guardrail is unscorable here"

    console = Console(record=True, width=140)
    render_evaluation(result, console)
    text = console.export_text()
    for name in unmeasured:
        row = next(line for line in text.splitlines() if name in line)
        assert "0.000" not in row and "| 0 " not in row, row

    markdown = as_markdown(result)
    row = next(line for line in markdown.splitlines() if unmeasured[0] in line)
    assert "—" in row

    import json

    payload = json.loads(as_json(result))
    for name in unmeasured:
        assert payload["metrics"][name]["point"] is None
        assert payload["metrics"][name]["measured"] is False


# --- context retention discriminates (§11.5) ------------------------------


async def test_the_retention_curve_separates_a_forgetful_target(tmp_path: Path) -> None:
    """A curve that is flat regardless of the target measures nothing."""
    remembers = await _evaluate(
        "openai_clean", tmp_path / "a", key="test-key-abcdefgh", profile="standard"
    )
    forgets = await _evaluate("drops_context_at_8", tmp_path / "b", profile="standard")

    assert remembers.retention_curve[15] > forgets.retention_curve[15]
    assert (
        remembers.metrics["context_retention_auc"].point
        > forgets.metrics["context_retention_auc"].point
    )


async def test_the_auc_weights_are_published(tmp_path: Path) -> None:
    """§11.5: depth spacing sets them, and `deep` changes them — which is why
    profile is a hard comparability key."""
    result = await _evaluate(
        "openai_clean", tmp_path, key="test-key-abcdefgh", profile="standard"
    )
    weights = result.retention_weights
    assert set(weights) == {3, 8, 15}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights[8] > weights[3], "the middle depth carries the most area"


async def test_a_family_ruled_out_by_capabilities_is_not_executed(
    tmp_path: Path,
) -> None:
    """Running probes for a SKIPPED family spends real money producing rows
    the report says do not exist."""
    result = await _evaluate("weird_shape", tmp_path, profile="standard")
    for family in result.families_not_run:
        assert family in {f for f, _ in result.skipped}
        assert not [o for o in result.observations if o.family == family]
