"""Every CLI verb, exercised as a command (spec §15, §16).

Four verbs had no CLI-level test of any kind -- `discover`, `init`, `version`
and `mock serve` -- and `baseline`/`gate` were tested only as library
functions. That last gap is the serious one: `gate`'s whole contract in CI is
its **exit code**, and no test had ever produced one through the command that
CI actually runs. Argument parsing, `--gate-on`, `--min-effect`, `--format`
and the exit path were all unexercised.

These use the mock, so they cost nothing. What they cover is the plumbing
between the command line and the engine, which is where a verb can be
completely broken while every library test passes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.conftest import make_app, make_client
from typer.testing import CliRunner

from sweepeval.cli.main import app

runner = CliRunner()


# --- version --------------------------------------------------------------


def test_version_prints_the_installed_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    printed = result.output.strip()
    assert printed and printed[0].isdigit(), printed


def test_version_matches_the_packaged_metadata() -> None:
    """It reports the *installed* distribution, not a literal in the source --
    which is why an editable install that has gone stale reports the old
    number. Worth pinning so the two cannot silently diverge."""
    from sweepeval.schema.versions import TOOL_VERSION

    assert runner.invoke(app, ["version"]).output.strip() == TOOL_VERSION


# --- init -----------------------------------------------------------------


def test_init_adds_the_artifact_root_to_gitignore(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert ".sweepeval/" in (tmp_path / ".gitignore").read_text(encoding="utf-8")


def test_init_is_safe_to_run_twice(tmp_path, monkeypatch) -> None:
    """It writes into a file the user owns, so a second run must not duplicate
    the entry or clobber what is already there."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".gitignore").write_text("*.pyc\n", encoding="utf-8")
    runner.invoke(app, ["init"])
    runner.invoke(app, ["init"])
    content = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert content.count(".sweepeval/") == 1
    assert "*.pyc" in content, "init overwrote the user's own entries"


def test_init_says_what_is_safe_to_commit(tmp_path, monkeypatch) -> None:
    """§6.6. `baseline.json` is designed to be committed and the user has no
    way to know that unless told."""
    monkeypatch.chdir(tmp_path)
    output = runner.invoke(app, ["init"]).output
    assert "baseline.json" in output
    assert "safe to commit" in output.lower()


# --- discover -------------------------------------------------------------


@pytest.fixture
def mocked_discovery(monkeypatch):
    """Route the command's own client at the in-process mock.

    `discover_command` builds its own `httpx.AsyncClient`, so there is no
    injection point; the wrapper swaps the client and leaves the real
    discovery to run against the mock.
    """
    import sweepeval.cli.discover as cli_discover

    app_mock = make_app("openai_clean")
    real = cli_discover.discover_target

    async def patched(_client, url, key, **kwargs):
        client = make_client(app_mock)
        try:
            return await real(client, url, key, **kwargs)
        finally:
            await client.aclose()

    monkeypatch.setattr(cli_discover, "discover_target", patched)
    return app_mock


def test_discover_probes_and_reports_the_shape(tmp_path: Path, mocked_discovery) -> None:
    result = runner.invoke(
        app,
        ["discover", "https://mock.test/v1/chat/completions",
         "--key", "test-key-abcdefgh", "--root", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert "openai" in result.output.lower()


def test_discover_writes_an_editable_config(tmp_path: Path, mocked_discovery) -> None:
    """The emitted config is the documented way to correct a wrong guess, so
    `--out` writing nothing would strand the user."""
    out = tmp_path / "sweepeval.yaml"
    result = runner.invoke(
        app,
        ["discover", "https://mock.test/v1/chat/completions",
         "--key", "test-key-abcdefgh", "--root", str(tmp_path), "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "url" in out.read_text(encoding="utf-8")


def test_discover_reports_its_own_spend(tmp_path: Path, mocked_discovery) -> None:
    """§8.3's budget is the promise that looking costs at most 25 requests."""
    result = runner.invoke(
        app,
        ["discover", "https://mock.test/v1/chat/completions",
         "--key", "test-key-abcdefgh", "--root", str(tmp_path)],
    )
    assert "POST" in result.output


# --- mock serve -----------------------------------------------------------


def test_mock_serve_rejects_an_unknown_scenario() -> None:
    """The failure a typo produces. Starting a server on a scenario that does
    not exist would look like the mock working."""
    result = runner.invoke(app, ["mock", "serve", "--scenario", "no_such_scenario"])
    assert result.exit_code != 0
    assert "no_such_scenario" in result.output


def test_mock_serve_resolves_a_bundled_scenario() -> None:
    from sweepeval.cli.mock import find_scenario

    assert find_scenario("openai_clean") is not None
    assert find_scenario("definitely_not_a_scenario") is None


# --- baseline and gate, as commands ---------------------------------------


def _sweep_backed(monkeypatch, scenario: str):
    """Point the evaluate/baseline/gate path at the mock."""
    import sweepeval.cli.evaluate as cli_evaluate

    app_mock = make_app(scenario)
    real = cli_evaluate.aevaluate_target

    async def patched(url, **kwargs):
        client = make_client(app_mock)
        kwargs["client"] = client
        kwargs["authorization_prompt"] = False
        try:
            return await real(url, **kwargs)
        finally:
            await client.aclose()

    monkeypatch.setattr(cli_evaluate, "aevaluate_target", patched)


def test_baseline_writes_a_committable_file(tmp_path: Path, monkeypatch) -> None:
    _sweep_backed(monkeypatch, "openai_clean")
    out = tmp_path / "baseline.json"
    result = runner.invoke(
        app,
        ["baseline", "https://mock.test/v1/chat/completions",
         "--key", "test-key-abcdefgh", "--out", str(out),
         "--root", str(tmp_path), "--profile", "standard", "-n", "2",
         "--i-am-authorized", "--yes"],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["clusters"], "a baseline without clusters cannot be gated against"
    assert "test-key-abcdefgh" not in out.read_text(encoding="utf-8")


def test_gate_exits_zero_when_nothing_changed(tmp_path: Path, monkeypatch) -> None:
    """The exit code is the entire contract in CI, and no test had produced
    one through the command before."""
    _sweep_backed(monkeypatch, "openai_clean")
    out = tmp_path / "baseline.json"
    runner.invoke(
        app,
        ["baseline", "https://mock.test/v1/chat/completions",
         "--key", "test-key-abcdefgh", "--out", str(out), "--root", str(tmp_path),
         "--profile", "standard", "-n", "2", "--i-am-authorized", "--yes"],
    )
    result = runner.invoke(
        app,
        ["gate", "https://mock.test/v1/chat/completions",
         "--key", "test-key-abcdefgh", "--baseline", str(out),
         "--root", str(tmp_path), "--profile", "standard", "-n", "2",
         "--i-am-authorized", "--yes"],
    )
    assert result.exit_code == 0, result.output


def test_gate_refuses_a_quick_profile_with_the_usage_code(
    tmp_path: Path, monkeypatch
) -> None:
    """§16: quick is not gate-eligible. Exit 3 is usage error, distinct from
    1 (regressed) and 2 (incomparable) -- a CI job branches on which.

    Both sides are quick on purpose. A standard baseline against a quick run
    refuses as *incomparable* first, with exit 2, because `profile` is a hard
    comparability key -- correct, but a different refusal from this one.
    """
    _sweep_backed(monkeypatch, "openai_clean")
    out = tmp_path / "baseline.json"
    runner.invoke(
        app,
        ["baseline", "https://mock.test/v1/chat/completions",
         "--key", "test-key-abcdefgh", "--out", str(out), "--root", str(tmp_path),
         "--profile", "quick", "-n", "2", "--i-am-authorized", "--yes"],
    )
    result = runner.invoke(
        app,
        ["gate", "https://mock.test/v1/chat/completions",
         "--key", "test-key-abcdefgh", "--baseline", str(out),
         "--root", str(tmp_path), "--profile", "quick", "-n", "2",
         "--i-am-authorized", "--yes"],
    )
    assert result.exit_code == 3, result.output


def test_gate_rejects_an_unknown_metric_rather_than_gating_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    """A typo in --gate-on used to gate nothing, silently, and exit 0 on a run
    whose security rate had gone to zero."""
    _sweep_backed(monkeypatch, "openai_clean")
    out = tmp_path / "baseline.json"
    runner.invoke(
        app,
        ["baseline", "https://mock.test/v1/chat/completions",
         "--key", "test-key-abcdefgh", "--out", str(out), "--root", str(tmp_path),
         "--profile", "standard", "-n", "2", "--i-am-authorized", "--yes"],
    )
    result = runner.invoke(
        app,
        ["gate", "https://mock.test/v1/chat/completions",
         "--key", "test-key-abcdefgh", "--baseline", str(out),
         "--root", str(tmp_path), "--profile", "standard", "-n", "2",
         "--gate-on", "secuirty_pass_rate", "--i-am-authorized", "--yes"],
    )
    # Exit 3 (usage), not a traceback: the message naming the metric and the
    # available ones is the whole value of the check, and it escaped as an
    # unhandled ValueError until a CLI-level test looked.
    assert result.exit_code == 3, result.output
    assert "secuirty_pass_rate" in result.output
    assert "security_pass_rate" in result.output, "the correction is not offered"


def test_gate_writes_a_machine_readable_payload(tmp_path: Path, monkeypatch) -> None:
    _sweep_backed(monkeypatch, "openai_clean")
    out = tmp_path / "baseline.json"
    runner.invoke(
        app,
        ["baseline", "https://mock.test/v1/chat/completions",
         "--key", "test-key-abcdefgh", "--out", str(out), "--root", str(tmp_path),
         "--profile", "standard", "-n", "2", "--i-am-authorized", "--yes"],
    )
    result = runner.invoke(
        app,
        ["gate", "https://mock.test/v1/chat/completions",
         "--key", "test-key-abcdefgh", "--baseline", str(out),
         "--root", str(tmp_path), "--profile", "standard", "-n", "2",
         "--format", "json", "--i-am-authorized", "--yes"],
    )
    assert result.exit_code == 0, result.output

    payloads = list(Path(tmp_path).rglob("gate.json"))
    assert payloads, "--format json wrote no gate.json"
    body = json.loads(payloads[0].read_text(encoding="utf-8"))
    for field in ("ok", "exit_code", "metrics", "not_gated", "incomplete"):
        assert field in body, field
