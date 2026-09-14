"""`docs/concepts/profiles.md` publishes the numbers a user consents to.

Every one of them had drifted. The table said 62 calls per run at `quick` and
191 at `standard` against a corpus that had moved to 60 and 188, so the request
totals were wrong too -- and the wall-clock column was computed by dividing by
a concurrency of 2 that the executor never dispatched, which halved it again.
A reader planning a spend was reading four stale numbers and one fictional one.

Nothing checked them, which is the whole reason. The docs suite verified that
commands run and that flags exist; the figures were prose to it. They are the
part a user acts on before paying, so they are the part that has to be tied to
the code that produces them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from sweepeval.corpus.loader import load_corpus
from sweepeval.execute.budget import estimate_run
from sweepeval.http.governor import DEFAULT_CONCURRENCY

DOC = Path("docs/concepts/profiles.md")
ROW = re.compile(
    r"^\|\s*`(?P<profile>quick|standard|deep)`\s*\|"
    r"\s*(?P<units>[\d,]+)\s*\|"
    r"\s*(?P<calls>[\d,]+)\s*\|"
    r"\s*(?P<configs>[\d,]+)\s*\|"
    r"\s*(?P<requests>[\d,]+|more)\s*\|"
    r"\s*(?P<wall>[^|]+)\|",
    re.M,
)


def _rows() -> dict[str, dict[str, str]]:
    found = {
        m.group("profile"): m.groupdict()
        for m in ROW.finditer(DOC.read_text(encoding="utf-8"))
    }
    assert set(found) == {"quick", "standard", "deep"}, sorted(found)
    return found


def _int(text: str) -> int:
    return int(text.replace(",", ""))


@pytest.mark.parametrize("profile", ["quick", "standard", "deep"])
def test_the_corpus_columns_are_the_corpus(profile: str) -> None:
    row = _rows()[profile]
    corpus = load_corpus(profile=profile)
    assert _int(row["units"]) == len(corpus.probes)
    assert _int(row["calls"]) == corpus.calls_per_run


@pytest.mark.parametrize(("profile", "configs"), [("quick", 6), ("standard", 12)])
def test_the_request_total_is_the_estimate(profile: str, configs: int) -> None:
    row = _rows()[profile]
    assert _int(row["configs"]) == configs
    estimate = estimate_run(
        load_corpus(profile=profile), configs=configs, runs=3, profile=profile
    )
    assert _int(row["requests"]) == estimate.total_requests


@pytest.mark.parametrize(("profile", "configs"), [("quick", 6), ("standard", 12)])
def test_the_wall_clock_is_the_estimate(profile: str, configs: int) -> None:
    """The column that was wrong by 2x, because it divided by a concurrency
    nothing dispatched."""
    row = _rows()[profile]
    estimate = estimate_run(
        load_corpus(profile=profile), configs=configs, runs=3, profile=profile
    )
    published = _int(re.search(r"(\d[\d,]*)", row["wall"]).group(1))
    assert published == round(estimate.wall_clock_minutes), (
        f"{profile}: doc says ~{published} min, estimator says "
        f"{estimate.wall_clock_minutes:.0f}"
    )


def test_the_table_header_does_not_promise_parallelism() -> None:
    """It read "Wall-clock at concurrency 2". The prose below the table may
    still say the number, because it explains what went wrong."""
    header = next(
        line
        for line in DOC.read_text(encoding="utf-8").splitlines()
        if line.startswith("| Profile |")
    )
    assert "concurrency" not in header.lower(), header


def test_implementing_concurrency_forces_this_page_to_be_rewritten() -> None:
    """The wall-clock column and the "one at a time" paragraph are both only
    true while dispatch is serial."""
    assert DEFAULT_CONCURRENCY == 1
    assert "one at a time" in DOC.read_text(encoding="utf-8")
