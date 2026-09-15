"""`--format` is a table now, and a typo in it is an error.

Two problems in one if/elif chain. The set of valid formats existed in three
places that could disagree -- the chain, the `--format` help text and the docs
-- and an unrecognised name printed a yellow line and carried on. So
`sweepeval report run --format htlm` wrote no file and exited 0: in CI, a
report step that passes by doing nothing. This tool refuses silent no-ops
everywhere else; I5 is the whole reason `SKIPPED` exists as a verdict.

I10 says adding a reporter touches no runner code, and it survived on a
technicality -- a CLI if/elif is not runner code. The extension point it
implies still did not exist, and nothing could read the list of formats to
print it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.conftest import cli_help
from typer.testing import CliRunner

from sweepeval.cli.main import app
from sweepeval.cli.report import REPORTERS

runner = CliRunner()

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "runs" / "demo-quick"

# The file each format is expected to leave behind. `terminal` writes none.
ARTEFACTS = {
    "html": "report.html",
    "junit": "report.xml",
    "json": "frontier.json",
    "md": "report.md",
}


def test_every_registered_format_is_covered_here() -> None:
    """So a format added to the registry cannot skip these tests."""
    assert set(REPORTERS) == {"terminal", *ARTEFACTS}


@pytest.mark.parametrize("name", sorted(ARTEFACTS))
def test_each_format_writes_its_file(name: str, tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["report", str(EXAMPLE), "--format", name, "--out", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    written = tmp_path / ARTEFACTS[name]
    assert written.is_file(), sorted(p.name for p in tmp_path.iterdir())
    assert written.stat().st_size > 0


def test_terminal_writes_nothing_and_still_succeeds(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["report", str(EXAMPLE), "--format", "terminal", "--out", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert not list(tmp_path.iterdir())


def test_several_formats_at_once_write_all_of_them(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["report", str(EXAMPLE), "--format", "md,json", "--out", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "report.md").is_file()
    assert (tmp_path / "frontier.json").is_file()


# --- a typo is an error, not a shrug ---------------------------------------


def test_an_unknown_format_fails(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["report", str(EXAMPLE), "--format", "htlm", "--out", str(tmp_path)]
    )
    assert result.exit_code == 2, result.output
    assert not list(tmp_path.iterdir()), "wrote something for a format it rejected"


def test_the_error_lists_what_it_would_have_accepted(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["report", str(EXAMPLE), "--format", "htlm", "--out", str(tmp_path)]
    )
    output = result.output.replace("\n", "")
    assert "htlm" in output
    for name in REPORTERS:
        assert name in output, f"{name} missing from the error's list"


def test_one_bad_name_rejects_the_whole_request(tmp_path: Path) -> None:
    """Writing the good half and failing on the rest leaves a caller with a
    partial report and a non-zero exit, which is the worst of both."""
    result = runner.invoke(
        app, ["report", str(EXAMPLE), "--format", "md,htlm", "--out", str(tmp_path)]
    )
    assert result.exit_code == 2, result.output
    assert not list(tmp_path.iterdir())


def test_a_bad_format_is_caught_before_the_run_is_even_read(tmp_path: Path) -> None:
    """Ranking a large run to then report that the output format was
    misspelled is a slow way to be told. The run directory here does not
    exist, and the format error is still what comes back."""
    result = runner.invoke(
        app, ["report", str(tmp_path / "nope"), "--format", "htlm"]
    )
    assert result.exit_code == 2
    assert "htlm" in result.output.replace("\n", "")


# --- the registry is the only list -----------------------------------------


def test_the_help_text_is_generated_from_the_registry() -> None:
    """It used to be a hand-written string, free to drift from the chain.

    Through `cli_help`, which strips rich's styling: with colour on the escape
    sequences land inside the rendered token, and this would otherwise have
    been the second assertion in the suite that passed locally and failed in
    CI for it.
    """
    help_text = cli_help("report").replace("\n", "").replace(" ", "")
    assert ",".join(sorted(REPORTERS)) in help_text, help_text
