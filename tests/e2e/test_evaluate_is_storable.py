"""A single-config run has to be re-reportable (I7, §5.1, §6.2).

I7: "Derived artifacts are regenerable from [the raw logs] without contacting
the endpoint." `sweepeval evaluate` wrote no `aggregates.json` at all, so its
run directory held only `calls.jsonl`, `observations.jsonl`, `blobs/` and
`state/` -- and `sweepeval report` on it raised

    no aggregates.json in ... - a run can only be re-reported if it was stored

The regenerability half of I7 was simply false for every single-config run,
and that is the *modal* outcome: §12.1 says a custom agent system typically
exposes no model list, no sampling parameters and no system-prompt slot, so
there is nothing to sweep and `evaluate` is the whole product.

The audit found it by listing the directory. Nothing had ever tried to reload
one.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from tests.conftest import make_app, make_client

from sweepeval.execute.evaluate import aevaluate_target
from sweepeval.report.stored import load_run, write_evaluation


async def _evaluate(scenario: str, root: Path):
    app = make_app(scenario)
    client = make_client(app)
    try:
        return await aevaluate_target(
            "https://mock.test" + app.scenario.paths[0],
            key="test-key-abcdefgh", client=client, root=str(root), runs=2,
            authorized=True, authorization_prompt=False, seed=7,
        )
    finally:
        await client.aclose()


@pytest.fixture(scope="module")
def stored():
    """One evaluation, stored the way the CLI stores it. Shared: slow."""
    import asyncio

    async def run():
        result = await _evaluate("openai_clean", Path(tempfile.mkdtemp()))
        write_evaluation(result)
        return result

    return asyncio.run(run())


def test_the_run_directory_holds_what_a_re_report_needs(stored) -> None:
    present = {p.name for p in Path(stored.store.run_dir).iterdir()}
    assert "aggregates.json" in present, present


def test_it_reloads_offline_with_no_key_and_no_network(stored) -> None:
    run = load_run(stored.store.run_dir)
    assert run.run_id == stored.run_id
    assert len(run.configs) == 1


def test_the_reloaded_numbers_are_the_ones_that_were_measured(stored) -> None:
    """Storing something re-readable is not the point; storing the *run* is."""
    run = load_run(stored.store.run_dir)
    row = run.configs[0]
    assert row.metrics, "reloaded with no metrics"
    for name, value in stored.metrics.items():
        assert row.metrics[name].point == pytest.approx(value.point), name


def test_the_cluster_tables_survive_so_it_can_be_re_ranked(stored) -> None:
    """Without them a stored run can be re-reported but never re-ranked, and
    `--prefer` and `--objectives` quietly need the network again."""
    row = load_run(stored.store.run_dir).configs[0]
    assert row.clusters
    assert row.clusters.get("security_pass_rate")


def test_it_carries_provenance_like_every_other_derived_artifact(stored) -> None:
    run = load_run(stored.store.run_dir)
    assert set(run.provenance) == {"calls.jsonl", "observations.jsonl"}
    assert run.stale is False


async def test_a_hard_failed_evaluation_reports_its_leaks_after_reload() -> None:
    """The leaks are classified in `execute` and carried on the result. An
    earlier draft reclassified them inside the reporter, which made `report`
    import `execute` and pulled the whole request layer in behind it -- the
    layering contract caught it, correctly."""
    result = await _evaluate("leaky_guardrails", Path(tempfile.mkdtemp()))
    write_evaluation(result)

    row = load_run(result.store.run_dir).configs[0]
    assert row.hard_fails.count == len(result.hard_fails)
    assert row.hard_fails.count > 0, "the leaky scenario stopped leaking"
    for hard in row.hard_fails.confirmed:
        assert hard.attack_class and hard.reason
