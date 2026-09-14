"""The two reasons `aggregates.json` is absent take opposite advice.

A directory with no run in it was never stored: re-run somewhere readable. A
directory with calls and no aggregates is a run still in progress or
interrupted -- aggregates are written once, at the end -- and re-running is the
worst thing the user could do.

One message answered both, with the first answer. Peeking at a live 7,981-
request sweep from a second terminal printed "a run can only be re-reported if
it was stored; re-run with a --root you can read back", whose obvious response
is to abandon several thousand paid requests and buy them again. §3 makes cost
a first-class constraint, so an error that recommends spending has to be right
about it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sweepeval.report.stored import load_run


def _run_dir(tmp_path: Path, name: str) -> Path:
    directory = tmp_path / name
    directory.mkdir()
    return directory


def test_a_directory_with_no_run_in_it_says_re_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as caught:
        load_run(_run_dir(tmp_path, "20260914T000000000-aaaaaa"))
    message = str(caught.value)
    assert "re-run with a --root" in message
    assert "resume" not in message


def test_a_run_with_calls_and_no_aggregates_says_resume(tmp_path: Path) -> None:
    directory = _run_dir(tmp_path, "20260914T101858086-6ea4f9")
    (directory / "calls.jsonl").write_text('{"ts": "x"}\n', encoding="utf-8")

    with pytest.raises(FileNotFoundError) as caught:
        load_run(directory)
    message = str(caught.value)

    assert "still running or was interrupted" in message
    assert "Do not start a new run" in message
    assert "--resume 20260914T101858086-6ea4f9" in message, (
        "advice that names no run id is advice the user cannot follow"
    )
    assert "re-run with a --root" not in message, (
        "the expensive advice must not survive in the in-progress branch"
    )


def test_an_empty_calls_file_is_not_a_run_in_progress(tmp_path: Path) -> None:
    """A store that was created and never written to is the first case, not
    the second: there is nothing to resume and nothing paid for."""
    directory = _run_dir(tmp_path, "20260914T000000000-bbbbbb")
    (directory / "calls.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(FileNotFoundError) as caught:
        load_run(directory)
    assert "re-run with a --root" in str(caught.value)
