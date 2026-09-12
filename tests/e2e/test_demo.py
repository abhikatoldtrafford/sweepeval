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
