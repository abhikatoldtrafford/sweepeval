"""I7: a derived artifact names the logs it was derived from.

The invariant table gives I7's enforcement mechanism as "derived files carry a
`derived_from` hash". `store/derived.py` implemented exactly that, complete
with a staleness check -- and every production writer went through
`write_json` instead, so no `aggregates.json` or `frontier.json` the tool had
ever written carried a `derived_from` at all. The only callers were the
module's own unit tests, which passed.

That is the failure mode the whole codebase keeps producing: a mechanism that
looks correct, has tests, and is not connected to anything. A check nothing
routes through is not a check.

The consequence is specific. `sweepeval report` re-reports a stored run
offline, and an `aggregates.json` computed before the log it sits beside grew
is a *subset* of the run presented as the run. Nothing distinguished the two.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.conftest import make_app, make_client
from typer.testing import CliRunner

from sweepeval.cli.main import app
from sweepeval.report.stored import load_run
from sweepeval.store.derived import provenance_of, read_payload, write_derived

LOGS = {"calls.jsonl", "observations.jsonl"}


@pytest.fixture(scope="module")
def swept(tmp_path_factory) -> Path:
    """One real sweep through the CLI, written to disk. Shared: it is slow.

    Through the CLI because `write_aggregates` is a CLI-level step -- which is
    also why routing it through `write_derived` had to happen there and not in
    the sweep engine.
    """
    import asyncio

    from sweepeval.cli import sweep as cli_sweep

    root = tmp_path_factory.mktemp("prov")
    mock = make_app("openai_clean")
    client = make_client(mock)
    original = cli_sweep.asweep_target

    async def patched(url, **kwargs):
        kwargs["client"] = client
        kwargs["authorization_prompt"] = False
        kwargs["config_cap"] = 2
        return await original(url, **kwargs)

    cli_sweep.asweep_target = patched  # type: ignore[assignment]
    try:
        result = CliRunner().invoke(
            app,
            [
                "sweep", "https://mock.test/v1/chat/completions",
                "--key", "test-key-abcdefgh", "--root", str(root),
                "--yes", "--i-am-authorized", "-n", "2", "--seed", "7",
                "--format", "json",
            ],
        )
    finally:
        cli_sweep.asweep_target = original  # type: ignore[assignment]
        asyncio.run(client.aclose())

    assert result.exit_code == 0, result.output
    return sorted((root / "runs").iterdir())[0]


# --- the provenance is written at all -------------------------------------


@pytest.mark.parametrize("name", ["aggregates.json", "frontier.json"])
def test_a_derived_artifact_records_which_logs_it_came_from(
    swept: Path, name: str
) -> None:
    payload, provenance, _ = read_payload(swept / name)
    assert payload, f"{name} has no payload"
    assert set(provenance) == LOGS, provenance
    assert all(len(h) == 64 for h in provenance.values()), provenance


@pytest.mark.parametrize("name", ["aggregates.json", "frontier.json"])
def test_the_recorded_hashes_are_the_logs_actually_on_disk(
    swept: Path, name: str
) -> None:
    """A provenance map of the right shape and the wrong contents would pass
    the test above and detect nothing."""
    _payload, provenance, _ = read_payload(swept / name)
    assert provenance == provenance_of(swept)


# --- and it detects the thing it exists to detect -------------------------


def test_a_grown_log_makes_the_stored_aggregates_stale(swept: Path, tmp_path) -> None:
    """The case in the docstring: report an aggregates.json whose observations
    log has since grown, and it is a subset of the run presented as the run."""
    copy = tmp_path / "grown"
    copy.mkdir()
    for item in swept.iterdir():
        if item.is_file():
            (copy / item.name).write_bytes(item.read_bytes())

    assert load_run(copy).stale is False

    with (copy / "observations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"metric": "later"}) + "\n")

    assert load_run(copy).stale is True


def test_the_cli_says_so_rather_than_reporting_the_subset(
    swept: Path, tmp_path
) -> None:
    """I7 makes derived files regenerable, so the answer is to regenerate --
    but silently reporting the stale numbers is not that."""
    copy = tmp_path / "cli"
    copy.mkdir()
    for item in swept.iterdir():
        if item.is_file():
            (copy / item.name).write_bytes(item.read_bytes())

    clean = CliRunner().invoke(app, ["report", str(copy), "--format", "terminal"])
    assert "stale" not in clean.output

    with (copy / "calls.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"call_id": "later"}) + "\n")

    grown = CliRunner().invoke(app, ["report", str(copy), "--format", "terminal"])
    assert "stale" in grown.output
    assert grown.exit_code == 0, "stale is a warning, not a failure"


# --- reading what was written before the envelope existed -----------------


def test_a_run_written_before_the_envelope_still_re_reports(
    swept: Path, tmp_path
) -> None:
    """Every run stored so far -- including the published OpenAI scorecard --
    is a flat payload. Refusing those would make an audit-trail change break
    offline re-reporting of exactly the runs an audit trail is for."""
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    for item in swept.iterdir():
        if item.is_file():
            (legacy / item.name).write_bytes(item.read_bytes())

    payload, _, _ = read_payload(legacy / "aggregates.json")
    (legacy / "aggregates.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    run = load_run(legacy)
    assert run.configs, "a legacy run must still load"
    assert run.provenance == {}
    assert run.stale is None, "unknown must not be reported as fresh"


def test_a_flat_payload_is_not_mistaken_for_an_envelope(tmp_path) -> None:
    """`payload` alone is a plausible field name for a future flat document,
    and unwrapping one would silently return a fragment of it."""
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"payload": {"a": 1}, "b": 2}), encoding="utf-8")
    body, provenance, _ = read_payload(path)
    assert body == {"payload": {"a": 1}, "b": 2}
    assert provenance == {}


def test_staleness_is_undecidable_without_a_current_map(tmp_path) -> None:
    path = tmp_path / "y.json"
    write_derived(path, {"a": 1}, derived_from={"calls.jsonl": "abc"})
    _body, _provenance, stale = read_payload(path)
    assert stale is None
