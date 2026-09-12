"""`sweepeval demo` (spec §20, D33)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from sweepeval.cli.main import app
from sweepeval.cli.mock import find_scenario

runner = CliRunner()


def test_the_demo_scenario_ships_inside_the_package() -> None:
    """An installed user has no tests/ directory.

    A bundled scenario that lives only in the repo works in CI and fails for
    every user, which is the class of bug that is invisible until release.
    """
    import sweepeval

    path = find_scenario("demo")
    assert path is not None
    package = Path(sweepeval.__file__).resolve().parent
    assert package in path.resolve().parents


def test_the_demo_runs_the_whole_pipeline(tmp_path: Path) -> None:
    result = runner.invoke(app, ["demo", "--root", str(tmp_path), "--keep", "-n", "2"])
    assert result.exit_code == 0, result.output
    assert "frontier" in result.output
    assert (tmp_path / "runs").exists()


def test_the_demo_says_it_is_simulated_before_and_after(tmp_path: Path) -> None:
    """The mock is never presented as evidence."""
    result = runner.invoke(app, ["demo", "--root", str(tmp_path), "--keep", "-n", "2"])
    assert "simulated endpoint" in result.output
    assert "simulated target" in result.output


def test_the_demo_finds_real_axes(tmp_path: Path) -> None:
    """A demo that shows a single config demonstrates nothing about a sweep."""
    result = runner.invoke(app, ["demo", "--root", str(tmp_path), "--keep", "-n", "2"])
    assert "axes" in result.output
    assert "configurations" in result.output
    assert "cfg-01" in result.output


def test_the_demo_target_cannot_resolve_if_it_escaped(tmp_path: Path) -> None:
    """`.invalid` is reserved and never resolves, so a demo that somehow left
    the in-process transport fails rather than reaching a stranger."""
    from sweepeval.cli.demo import DEMO_URL

    assert DEMO_URL.split("/")[2].endswith(".invalid")


def test_run_calls_sweep_with_every_parameter_it_has() -> None:
    """`run` forwards to `sweep` by keyword, so a new sweep parameter without
    a default silently breaks `run` at runtime and nowhere else.

    This is not hypothetical: adding `cap` to the confirmer builder broke the
    sweep verb the same way, and no test caught it.
    """
    import inspect

    from sweepeval.cli.sweep import run_command, sweep_command

    sweep_params = set(inspect.signature(sweep_command).parameters)
    call = inspect.getsource(run_command)
    forwarded = {
        name for name in sweep_params if f"{name}=" in call
    }
    missing = sweep_params - forwarded
    assert not missing, f"run_command does not forward: {sorted(missing)}"


def test_run_is_invocable_end_to_end(tmp_path: Path) -> None:
    """The forwarding test above is static; this one actually calls it."""
    import sweepeval.cli.sweep as cli_sweep
    from sweepeval.cli.mock import find_scenario
    from sweepeval.mock.app import MockApp
    from sweepeval.mock.scenario import load_scenario

    scenario = find_scenario("demo")
    assert scenario is not None
    mock = MockApp(load_scenario(scenario))
    original = cli_sweep.asweep_target

    async def patched(url, **kwargs):
        import httpx

        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mock), base_url="https://demo.test"
        )
        kwargs["client"] = client
        kwargs["authorization_prompt"] = False
        kwargs["authorized"] = True
        kwargs["config_cap"] = 1
        try:
            return await original(url, **kwargs)
        finally:
            await client.aclose()

    cli_sweep.asweep_target = patched  # type: ignore[assignment]
    try:
        result = runner.invoke(
            app,
            ["run", "https://demo.test/v1/chat/completions",
             "--root", str(tmp_path), "--yes", "-n", "2"],
        )
    finally:
        cli_sweep.asweep_target = original  # type: ignore[assignment]

    assert result.exit_code == 0, result.output
    assert "frontier" in result.output
