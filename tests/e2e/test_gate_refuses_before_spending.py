"""An ineligible profile is refused before the gate spends anything (§3, I9).

`sweepeval gate --profile quick` exits 3, correctly: quick's cluster counts sit
just above the bootstrap floor, so its intervals are too wide to gate on, and
refusing beats passing everything and calling it a green build.

It used to refuse *after* re-running the target. The check lived inside
`gate()`, which is a pure function over a result that has already been
measured, so the command paid for a full evaluation and then declined to use
it. Run against api.openai.com the first time this path touched a real
endpoint, that was 120 requests spent to be told the profile on the command
line was the wrong one — knowable before the first byte went out.

§3 makes cost a first-class constraint and I9 puts the estimate before any
request of any kind. Charging for a refusal inverts both.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.conftest import make_app, make_client
from typer.testing import CliRunner

from sweepeval.cli.main import app
from sweepeval.execute.gate import profile_refusal

runner = CliRunner()


@pytest.fixture
def counting_target(monkeypatch):
    """Count every request the CLI sends.

    Patched on `sweepeval.cli.evaluate`, not on `sweepeval.execute.evaluate`:
    `_run` does `from ... import aevaluate_target`, so it holds its own
    reference and patching the source module intercepts nothing. The first
    version of this fixture did exactly that, and its request count was
    vacuously zero whether the fix was present or not.
    """
    import httpx

    import sweepeval.cli.evaluate as cli
    import sweepeval.execute.evaluate as ev

    sent = {"n": 0}
    real = ev.aevaluate_target

    async def counted(*args, **kwargs):
        inner = httpx.ASGITransport(app=make_app("openai_clean"))

        class Counting(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                sent["n"] += 1
                return await inner.handle_async_request(request)

        kwargs["client"] = httpx.AsyncClient(
            transport=Counting(), base_url="https://mock.test"
        )
        try:
            return await real(*args, **kwargs)
        finally:
            await kwargs["client"].aclose()

    monkeypatch.setattr(cli, "aevaluate_target", counted)
    return sent


def _gate(tmp_path: Path, baseline: Path, profile: str):
    return runner.invoke(
        app,
        ["gate", "https://mock.test/v1/chat/completions",
         "--key", "test-key-abcdefgh", "--baseline", str(baseline),
         "--root", str(tmp_path / "root"), "--profile", profile, "-n", "2",
         "--i-am-authorized", "--yes"],
    )


def _baseline(tmp_path: Path) -> Path:
    """A baseline file the gate can load. Its contents do not matter here —
    the refusal must happen before it is even consulted."""
    app_ = make_app("openai_clean")
    client = make_client(app_)
    import asyncio

    import sweepeval.execute.evaluate as ev
    from sweepeval.execute.gate import save_baseline, snapshot

    async def run():
        try:
            return await ev.aevaluate_target(
                "https://mock.test/v1/chat/completions",
                key="test-key-abcdefgh", client=client,
                root=str(tmp_path / "base"), runs=2, profile="quick",
                authorized=True, authorization_prompt=False, seed=7,
            )
        finally:
            await client.aclose()

    return save_baseline(snapshot(asyncio.run(run())), tmp_path / "baseline.json")


# --- the refusal is free ----------------------------------------------------


def test_a_quick_gate_sends_nothing(tmp_path: Path, counting_target) -> None:
    baseline = _baseline(tmp_path)
    counting_target["n"] = 0  # the baseline above is setup, not the gate

    result = _gate(tmp_path, baseline, "quick")

    assert result.exit_code == 3, result.output
    assert counting_target["n"] == 0, (
        f"refused after sending {counting_target['n']} request(s)"
    )


def test_it_still_says_why_and_what_to_do(tmp_path: Path) -> None:
    baseline = _baseline(tmp_path)
    result = _gate(tmp_path, baseline, "quick")
    output = result.output.replace("\n", " ")
    assert "not gate-eligible" in output
    assert "--profile standard" in output


def test_no_run_directory_is_created_for_a_refused_gate(
    tmp_path: Path, counting_target
) -> None:
    """Nothing was measured, so there is nothing to store. A run directory
    with no calls in it is the shape `report` reads as an interrupted sweep."""
    baseline = _baseline(tmp_path)
    _gate(tmp_path, baseline, "quick")
    assert not (tmp_path / "root").exists()


# --- and it is the same refusal, at the same code ---------------------------


def test_the_early_check_and_the_late_one_agree() -> None:
    """Two call sites for one rule. `gate()` keeps its own copy because it is
    a public pure function that callers reach without the CLI."""
    assert profile_refusal("quick") is not None
    assert profile_refusal("standard") is None
    assert profile_refusal("deep") is None


def test_an_eligible_profile_is_not_refused_early(tmp_path: Path) -> None:
    """Otherwise the cheapest way to pass this file is to refuse everything."""
    baseline = _baseline(tmp_path)
    result = _gate(tmp_path, baseline, "standard")
    assert result.exit_code != 3 or "not gate-eligible" not in result.output
