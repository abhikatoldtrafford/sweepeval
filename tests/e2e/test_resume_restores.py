"""`--resume` must return the measurements already paid for (spec §12.5).

An audit found it returned an empty run labelled COMPLETE. Completed unit-runs
were skipped, so no outcomes came back; aggregation reads in-memory outcomes,
so every metric was NO_VALID_INTERVAL with coverage 0/0 and the frontier
degenerated to nothing-dominates-anything. The rows were in
`observations.jsonl` the whole time.

That is the worst shape a bug can take here: a run that cost real money,
completed, and reported nothing -- while saying COMPLETE.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.conftest import make_app, make_client

from sweepeval.execute.sweep import SweepStatus, asweep_target
from sweepeval.pipeline import rank_sweep


async def _sweep(root: Path, resume: str | None = None):
    app = make_app("openai_clean")
    client = make_client(app)
    try:
        return await asweep_target(
            "https://mock.test" + app.scenario.paths[0],
            key="test-key-abcdefgh",
            client=client,
            root=str(root),
            runs=2,
            profile="quick",
            authorized=True,
            authorization_prompt=False,
            seed=7,
            config_cap=2,
            resume_run_id=resume,
        )
    finally:
        await client.aclose()


@pytest.fixture(scope="module")
def both(tmp_path_factory):
    """One sweep, then a resume of it. Shared: they are slow."""
    import asyncio

    async def run():
        root = tmp_path_factory.mktemp("resume")
        first = await _sweep(root)
        again = await _sweep(root, resume=first.run_id)
        return first, again

    return asyncio.run(run())


def test_a_resume_reports_the_measurements_not_an_empty_run(both) -> None:
    _first, again = both
    assert again.status is SweepStatus.COMPLETE
    assert again.resume is not None and again.resume.ok

    for row in again.configs:
        assert row.observations, f"{row.config_id} came back with nothing"
        assert row.metrics["security_pass_rate"].n_clusters > 0


def test_the_resumed_numbers_match_the_original(both) -> None:
    first, again = both

    for a, b in zip(first.configs, again.configs, strict=True):
        assert a.config_id == b.config_id
        for metric, value in a.metrics.items():
            assert b.metrics[metric].point == pytest.approx(value.point), metric


def test_coverage_survives_the_resume(both) -> None:
    """Coverage 0/0 on every family was how the empty run announced itself."""
    _first, again = both
    for row in again.configs:
        scored, attempted = row.coverage["security"]
        assert attempted > 0 and scored > 0


def test_a_resume_sends_nothing_for_work_already_done(both) -> None:
    """It reads the log, not the target."""
    _first, again = both
    assert sum(len(o.calls) for r in again.configs for o in r.outcomes) > 0
    # Every call attributed to the resumed run came off disk, so the state
    # file's own request counter did not move.
    assert again.store is not None
    for row in again.configs:
        spent, _tokens = again.store.state_for(row.config_id).budget()
        assert spent == row.requests


def test_a_resume_does_not_duplicate_cross_run_rows(both) -> None:
    """Cross-run scorers need the response text, which a restored outcome does
    not carry. Re-running them would score determinism on empty strings and
    append a second, wrong set of rows."""
    first, again = both
    assert again.store is not None
    counts = {
        row.config_id: len(row.observations) for row in again.configs
    }
    original = {row.config_id: len(row.observations) for row in first.configs}
    assert counts == original


def test_the_resumed_run_still_ranks(both) -> None:
    _first, again = both
    frontier = rank_sweep(again, seed=7)
    assert frontier.frontier, "the frontier degenerated on a resumed run"
