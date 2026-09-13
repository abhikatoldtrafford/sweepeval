"""Offline reporting, HTML/JUnit, and `compare` (spec §15, §6.5, D30, D33)."""

from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from tests.conftest import make_app, make_client
from typer.testing import CliRunner

import sweepeval.cli.sweep as cli_sweep
from sweepeval.cli.main import app
from sweepeval.report.compare import compare_runs
from sweepeval.report.html import as_html, as_junit
from sweepeval.report.stored import load_run

runner = CliRunner()


@pytest.fixture(scope="module")
def swept_dir(tmp_path_factory) -> Path:
    """One CLI sweep against the mock, stored on disk."""
    root = tmp_path_factory.mktemp("cli")
    app_mock = make_app("openai_clean")
    client = make_client(app_mock)
    original = cli_sweep.asweep_target

    async def patched(url, **kwargs):
        kwargs["client"] = client
        kwargs["authorization_prompt"] = False
        kwargs["config_cap"] = 2
        return await original(url, **kwargs)

    cli_sweep.asweep_target = patched  # type: ignore[assignment]
    try:
        result = runner.invoke(
            app,
            [
                "sweep", "https://mock.test/v1/chat/completions",
                "--key", "test-key-abcdefgh", "--root", str(root),
                "--yes", "--i-am-authorized", "-n", "2", "--seed", "7",
                "--format", "html,junit",
            ],
        )
    finally:
        cli_sweep.asweep_target = original  # type: ignore[assignment]
        import asyncio

        asyncio.run(client.aclose())

    assert result.exit_code == 0, result.output
    runs = sorted((root / "runs").iterdir())
    assert runs, result.output
    return runs[0]


def test_the_sweep_writes_every_artifact(swept_dir: Path) -> None:
    present = {p.name for p in swept_dir.iterdir()}
    assert {
        "manifest.json",
        "plan.json",
        "aggregates.json",
        "frontier.json",
        "calls.jsonl",
        "observations.jsonl",
        "report.html",
        "report.xml",
    } <= present, present


def test_a_stored_run_reloads_with_its_numbers(swept_dir: Path) -> None:
    run = load_run(swept_dir)
    assert run.configs
    for row in run.configs:
        assert row.metrics
        assert row.clusters, "without cluster tables a stored run cannot be re-ranked"


def test_re_ranking_offline_gives_the_same_frontier(swept_dir: Path) -> None:
    """D33: changing a preference needs no re-run, so the offline path has to
    reach the same answer as the live one."""
    from sweepeval.pipeline import rank_sweep
    from sweepeval.store.json_io import read_json

    stored = read_json(swept_dir / "frontier.json")
    # The same seed the sweep ranked with; a different one would compare two
    # different bootstraps and pass only because the frontier was degenerate.
    offline = rank_sweep(load_run(swept_dir), seed=7)
    assert list(offline.frontier) == stored["frontier"]
    assert {k: list(v) for k, v in offline.dominated.items()} == stored["dominated"]
    assert [list(c.members) for c in offline.clusters] == [
        c["members"] for c in stored["clusters"]
    ]


def test_the_report_verb_needs_no_network(swept_dir: Path) -> None:
    result = runner.invoke(
        app, ["report", str(swept_dir), "--format", "md", "--prefer", "security"]
    )
    assert result.exit_code == 0, result.output
    text = (swept_dir / "report.md").read_text(encoding="utf-8")
    assert "sweepeval" in text
    assert "frontier" in text


def test_the_report_verb_refuses_a_directory_with_no_run(tmp_path: Path) -> None:
    result = runner.invoke(app, ["report", str(tmp_path)])
    assert result.exit_code == 2
    assert "aggregates.json" in result.output


# --- HTML (D30) ------------------------------------------------------------


def test_the_html_page_is_self_contained(swept_dir: Path) -> None:
    """It has to render on a laptop with no internet."""
    html = (swept_dir / "report.html").read_text(encoding="utf-8")
    assert "<script" not in html
    assert "http://" not in html
    assert not re.search(r'src\s*=\s*"https?://', html)
    assert not re.search(r'<link[^>]+href\s*=\s*"https?://', html)


def test_the_html_page_draws_its_own_plots(swept_dir: Path) -> None:
    html = (swept_dir / "report.html").read_text(encoding="utf-8")
    assert "<svg" in html
    assert "matplotlib" not in html


def test_the_html_page_says_quick_is_not_gate_eligible(swept_dir: Path) -> None:
    html = (swept_dir / "report.html").read_text(encoding="utf-8")
    assert "not gate-eligible" in html
    # The wording that was true of the old 16-unit quick profile and is not
    # true of this one.
    assert "no valid intervals" not in html


def test_the_html_page_escapes_what_it_renders() -> None:
    from sweepeval.report.stored import StoredRun, _Corpus, _Plan, _Spec, _Status

    run = StoredRun(
        run_id="<script>alert(1)</script>",
        profile="quick",
        runs=1,
        status=_Status("COMPLETE"),
        corpus=_Corpus(1, 1),
        plan=_Plan(configs=(_Spec("c1", {}, "none", "x"),)),
    )
    html = as_html(run)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


# --- JUnit -----------------------------------------------------------------


def test_the_junit_document_parses_and_counts(swept_dir: Path) -> None:
    root = ET.fromstring((swept_dir / "report.xml").read_text(encoding="utf-8"))
    assert root.tag == "testsuite"
    assert int(root.get("tests", "0")) == len(root.findall("testcase"))


def test_a_metric_with_no_interval_is_skipped_not_passed() -> None:
    """I5: a green case would hide a metric nothing could be scored for."""
    from sweepeval.report.stored import (
        StoredConfig,
        StoredRun,
        _Corpus,
        _Plan,
        _Spec,
        _Status,
    )
    from sweepeval.schema.metric import CIMethod, Estimand, Flag, MetricValue

    spec = _Spec("c1", {}, "none", "c1")
    run = StoredRun(
        run_id="r", profile="quick", runs=1, status=_Status("COMPLETE"),
        corpus=_Corpus(1, 1), plan=_Plan(configs=(spec,)),
        configs=[
            StoredConfig(
                config=spec,
                metrics={
                    "guardrail_pass_rate": MetricValue(
                        point=0.0, method=CIMethod.none, n_clusters=0, alpha=0.05,
                        estimand=Estimand.generalization,
                        flags=(Flag.NO_VALID_INTERVAL,),
                    )
                },
            )
        ],
    )
    root = ET.fromstring(as_junit(run))
    case = root.find("testcase")
    assert case is not None
    assert case.find("skipped") is not None
    assert case.find("failure") is None


def test_a_config_that_never_ran_is_a_skipped_case() -> None:
    from sweepeval.report.stored import StoredRun, _Corpus, _Plan, _Spec, _Status

    run = StoredRun(
        run_id="r", profile="quick", runs=1, status=_Status("INCOMPLETE"),
        corpus=_Corpus(1, 1), plan=_Plan(configs=(_Spec("c1", {}, "none", "c1"),)),
        not_run=["cfg-09"],
    )
    root = ET.fromstring(as_junit(run))
    names = [c.get("name") for c in root.findall("testcase")]
    assert "config cfg-09" in names


# --- compare (I6) ----------------------------------------------------------


def test_two_copies_of_a_run_are_comparable(swept_dir: Path, tmp_path: Path) -> None:
    import shutil

    twin = tmp_path / "twin"
    shutil.copytree(swept_dir, twin)
    verdict = compare_runs(swept_dir, twin)
    assert verdict.ok, verdict.explain()


def test_a_changed_hard_key_refuses_and_names_it(
    swept_dir: Path, tmp_path: Path
) -> None:
    import json
    import shutil

    twin = tmp_path / "twin"
    shutil.copytree(swept_dir, twin)
    manifest = json.loads((twin / "manifest.json").read_text(encoding="utf-8"))
    manifest["comparability"]["hard"]["profile"] = "standard"
    (twin / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    verdict = compare_runs(swept_dir, twin)
    assert not verdict.ok
    assert any(m.key == "profile" for m in verdict.refusals)
    assert "quick" in verdict.explain() and "standard" in verdict.explain()


def test_the_compare_verb_exits_1_on_a_refusal(
    swept_dir: Path, tmp_path: Path
) -> None:
    """A refusal is a real outcome in CI, not a crash."""
    import json
    import shutil

    twin = tmp_path / "twin"
    shutil.copytree(swept_dir, twin)
    manifest = json.loads((twin / "manifest.json").read_text(encoding="utf-8"))
    manifest["comparability"]["hard"]["target_type"] = "agent"
    (twin / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = runner.invoke(app, ["compare", str(swept_dir), str(twin)])
    assert result.exit_code == 1
    assert "target type differs" in result.output

    same = runner.invoke(app, ["compare", str(swept_dir), str(swept_dir)])
    assert same.exit_code == 0


def test_compare_names_a_missing_manifest_rather_than_crashing(tmp_path: Path) -> None:
    result = runner.invoke(app, ["compare", str(tmp_path), str(tmp_path)])
    assert result.exit_code == 2
    assert "no readable manifest" in result.output


def test_offline_junit_keeps_a_confirmed_hard_fail() -> None:
    """`sweepeval report --format junit` showed a green CI tab on a run that
    had hard-failed, because the offline reader stored hard fails as prose and
    could only count them. A green CI tab is the one thing nobody re-checks.
    """
    from sweepeval.report.stored import (
        StoredConfig,
        StoredRun,
        _Corpus,
        _HardFails,
        _Plan,
        _RestoredHardFail,
        _Spec,
        _Status,
        aggregates_payload,
        load_run,
    )

    spec = _Spec("cfg-00", {}, "none", "cfg-00")
    leak = _RestoredHardFail(
        unit_id="sec.exfiltration.direct.v1",
        attack_class="system_prompt_exfiltration",
        reason="leaked on every one of 3 scored runs",
    )
    live = StoredRun(
        run_id="r", profile="quick", runs=3, status=_Status("COMPLETE"),
        corpus=_Corpus(1, 1), plan=_Plan(configs=(spec,)),
        configs=[StoredConfig(config=spec, hard_fails=_HardFails((leak,)))],
    )

    live_xml = ET.fromstring(as_junit(live))
    assert int(live_xml.get("failures", "0")) == 1

    # Round-trip it exactly as `report` does.
    payload = aggregates_payload(live)
    assert payload["configs"][0]["hard_fails"][0]["unit_id"] == leak.unit_id

    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        (directory / "aggregates.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        restored = load_run(directory)

    assert restored.configs[0].hard_fails.count == 1
    restored_xml = ET.fromstring(as_junit(restored))
    assert int(restored_xml.get("failures", "0")) == 1, (
        "the hard fail vanished on the round trip"
    )
    assert leak.unit_id in as_junit(restored)
