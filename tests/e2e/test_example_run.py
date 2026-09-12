"""The committed example run reproduces offline (spec §19.1, D33).

This is the README's evidence that a stored run is self-sufficient. If it stops
reproducing — because an artifact schema changed, or because reporting quietly
started needing something the example does not carry — the claim that
`--prefer` and `--objectives` need no re-run stops being true, and this is
where that shows up.

The run inside the example is against the bundled **mock**. That is stated in
`examples/README.md` and asserted here, because the one thing this fixture
must never become is evidence about a real target.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from sweepeval.cli.main import app
from sweepeval.pipeline import rank_sweep
from sweepeval.report.stored import load_run
from sweepeval.store.json_io import read_json

runner = CliRunner()

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "runs" / "demo-quick"


def test_the_example_run_is_committed() -> None:
    assert EXAMPLE.is_dir(), "the README's evidence is missing"
    present = {p.name for p in EXAMPLE.iterdir()}
    assert {"manifest.json", "plan.json", "aggregates.json", "frontier.json"} <= present


def test_it_reports_offline_with_no_key_and_no_endpoint(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["report", str(EXAMPLE), "--format", "md", "--out", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "sweepeval" in text
    assert "frontier" in text


def test_it_re_ranks_offline_from_its_stored_clusters() -> None:
    """Without the cluster tables a stored run could be re-reported but never
    re-ranked, and narrowing objectives would quietly need the target again."""
    run = load_run(EXAMPLE)
    assert run.configs
    for row in run.configs:
        assert row.clusters

    wide = rank_sweep(run, seed=0)
    narrow = rank_sweep(run, objectives=["security", "latency"], seed=0)
    assert len(wide.objectives) == 6
    assert len(narrow.objectives) == 2


def test_a_preference_can_be_applied_without_re_running(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "report", str(EXAMPLE), "--format", "terminal",
            "--prefer", "security,cost,latency",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "your preference" in result.output


def test_the_example_carries_no_credential() -> None:
    for path in EXAMPLE.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            assert "sk-" not in text, path
            assert "Bearer " not in text, path


def test_the_example_is_labelled_as_a_mock() -> None:
    """It must never be readable as evidence about a real endpoint."""
    readme = (EXAMPLE.parents[1] / "README.md").read_text(encoding="utf-8")
    assert "simulated" in readme.lower()
    assert "not as evidence" in " ".join(readme.split())

    target = read_json(EXAMPLE / "manifest.json")["target"]["url"]
    assert ".invalid" in target, target


def test_it_stays_small_enough_to_commit() -> None:
    """A multi-megabyte fixture is a fixture nobody clones happily."""
    total = sum(p.stat().st_size for p in EXAMPLE.rglob("*") if p.is_file())
    assert total < 1_000_000, f"{total} bytes"


@pytest.mark.parametrize("fmt", ["html", "junit"])
def test_every_reporter_renders_it(fmt: str, tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["report", str(EXAMPLE), "--format", fmt, "--out", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    written = list(tmp_path.iterdir())
    assert written, result.output
    assert written[0].stat().st_size > 0
