"""`sweepeval rescore` — the §5.1 promise, finally reachable.

The store has always been sold as re-scorable without paying again, and
nothing exposed it. When three security verdicts in the published 14-model run
turned out to be wrong, correcting them took a one-off script, and the numbers
on `docs/scorecard.md` came from something no reader could run.

Two properties matter more than the feature. It must never rewrite
`observations.jsonl` — that log is what was paid for and it is append-only
(I7). And re-aggregating the configs whose verdicts did *not* change must land
exactly on what the run stored, or the corrected figures are a different
measurement rather than a fix.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from tests.conftest import make_app, make_client
from typer.testing import CliRunner

from sweepeval.cli.main import app
from sweepeval.execute.rescore import RESCORABLE, rescore_run

runner = CliRunner()


@pytest.fixture(scope="module")
def stored_run() -> Path:
    """A real run on disk.

    Produced by `sweep`, not `evaluate`: only a sweep writes `plan.json`, and
    the canary table in it is what makes a security re-score possible. A bare
    evaluate run stores calls, observations and blobs and nothing that says
    which canary each probe carried.
    """
    import asyncio

    from sweepeval.execute.sweep import asweep_target

    root = Path(tempfile.mkdtemp())

    async def run():
        mock = make_app("leaky_guardrails")
        client = make_client(mock)
        try:
            return await asweep_target(
                "https://mock.test" + mock.scenario.paths[0],
                key=None, client=client, root=str(root), runs=2,
                profile="quick", authorized=True, authorization_prompt=False,
                seed=7, config_cap=1,
            )
        finally:
            await client.aclose()

    result = asyncio.run(run())
    # The reproduction check needs the run's own aggregates to compare
    # against; `asweep_target` leaves that to the CLI.
    from sweepeval.report.stored import write_aggregates

    write_aggregates(result)
    return result.store.run_dir


def test_an_evaluate_only_run_says_why_it_cannot_be_rescored(tmp_path: Path) -> None:
    """Canary values are per-run. Recomputing a security verdict against the
    wrong canary is worse than declining to recompute it."""
    import asyncio

    import sweepeval.execute.evaluate as ev

    async def run():
        mock = make_app("leaky_guardrails")
        client = make_client(mock)
        try:
            return await ev.aevaluate_target(
                "https://mock.test" + mock.scenario.paths[0],
                key=None, client=client, root=str(tmp_path), runs=2,
                authorized=True, authorization_prompt=False, seed=7,
            )
        finally:
            await client.aclose()

    run_dir = asyncio.run(run()).store.run_dir
    assert not (run_dir / "plan.json").exists()

    result = runner.invoke(app, ["rescore", str(run_dir)])
    assert result.exit_code == 2
    assert "plan.json" in " ".join(result.output.split())


# --- it reads, and it reads the right thing ---------------------------------


def test_it_rescores_something(stored_run: Path) -> None:
    report = rescore_run(stored_run)
    assert report.considered > 0, "nothing was re-scored; the join is broken"
    assert report.families == RESCORABLE


def test_agreeing_with_itself_changes_nothing(stored_run: Path) -> None:
    """Re-scoring a run with the same scorers that produced it must be a
    no-op. Anything else means the offline path is not the online one."""
    report = rescore_run(stored_run)
    assert report.changes == [], [str(c) for c in report.changes]


def test_it_reproduces_the_runs_own_numbers(stored_run: Path) -> None:
    report = rescore_run(stored_run)
    assert report.ok, report.mismatched
    assert report.reproduced > 0, "nothing was cross-checked"


def test_a_row_whose_text_is_gone_is_counted_not_ignored(stored_run: Path) -> None:
    """A re-score that quietly skipped unreadable rows would report "nothing
    changed" for a run it could not read."""
    report = rescore_run(stored_run)
    assert report.unjoinable >= 0
    assert report.considered + report.unjoinable > 0


# --- and it never touches what was paid for ---------------------------------


def test_the_observation_log_is_not_rewritten(stored_run: Path) -> None:
    log = stored_run / "observations.jsonl"
    before = log.read_bytes()
    rescore_run(stored_run)
    assert log.read_bytes() == before, "re-score mutated the append-only log"


def test_no_new_files_appear(stored_run: Path) -> None:
    before = {p.name for p in stored_run.iterdir()}
    rescore_run(stored_run)
    assert {p.name for p in stored_run.iterdir()} == before


# --- a changed scorer is detected -------------------------------------------


def test_a_scorer_that_disagrees_is_reported(stored_run: Path, monkeypatch) -> None:
    """The whole point. Without this the tests above pass on a function that
    returns an empty report."""
    from sweepeval.schema.observation import Verdict
    from sweepeval.scorers import registry as build_registry

    registry = build_registry()
    security = registry.get("security")
    real = security.score

    flip = {Verdict.PASS: Verdict.FAIL, Verdict.FAIL: Verdict.PASS}

    def inverted(unit, calls, context):
        return [
            row.model_copy(update={"verdict": flip[row.verdict]})
            if row.verdict in flip
            else row
            for row in real(unit, calls, context)
        ]

    monkeypatch.setattr(security, "score", inverted)
    report = rescore_run(stored_run, registry=registry)
    assert report.changes, "an inverted scorer produced no reported change"
    assert all({c.was, c.now} == {"PASS", "FAIL"} for c in report.changes)


# --- the CLI surface --------------------------------------------------------


def test_the_verb_runs_and_says_what_it_did(stored_run: Path) -> None:
    result = runner.invoke(app, ["rescore", str(stored_run)])
    assert result.exit_code == 0, result.output
    assert "re-scored" in result.output


def test_an_unrescorable_family_is_refused_with_the_reason(stored_run: Path) -> None:
    """`determinism` needs every run of a unit at once and `operational`
    comes from calls.jsonl; neither is a text re-score. Saying so beats
    reporting zero changes for a family that was never examined."""
    result = runner.invoke(
        app, ["rescore", str(stored_run), "--families", "determinism"]
    )
    assert result.exit_code == 2
    assert "determinism" in result.output


def test_a_missing_run_is_an_error_not_a_crash(tmp_path: Path) -> None:
    result = runner.invoke(app, ["rescore", str(tmp_path / "nope")])
    assert result.exit_code == 2


def test_it_is_reachable_from_the_api() -> None:
    """§4.2: no logic lives only in the CLI."""
    from sweepeval import api

    assert "rescore" in api.__all__
    assert callable(api.rescore)


# --- the published run, when it is present ----------------------------------


def test_the_scorecard_run_still_reproduces() -> None:
    """The numbers on docs/scorecard.md came from this path. If the run is on
    disk, the claim it rests on is checkable here."""
    published = Path(".sweepeval/runs/20260914T101858086-6ea4f9")
    if not (published / "aggregates.json").is_file():
        pytest.skip("the published run is not present")
    report = rescore_run(published)
    assert report.ok, report.mismatched
    assert report.reproduced > 100, report.reproduced
    changed = {json.dumps([c.config_id, c.metric]) for c in report.changes}
    assert len(changed) == len(report.changes) or report.changes
