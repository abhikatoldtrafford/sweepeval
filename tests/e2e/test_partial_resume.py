"""A resume that finishes a half-run config (spec §12.5, §11.4).

`tests/e2e/test_resume_restores.py` covers the *total* resume: nothing left to
run. That is the one case where the cross-run bug cannot happen, because the
guard keyed on `not executed` and took the restore path.

Interrupt mid-config and resume, and the guard does not fire. Restored
outcomes carried `text=""`; `_finalize_cross_run` marks a textless run
unscorable; so determinism was recomputed over empty strings. An audit
measured it: `target_determinism_at_temp0` 1.00 in an uninterrupted run, 0.00
after Ctrl-C plus `--resume` -- with a confidence interval, status COMPLETE,
and the wrong value propagated to every temperature sibling through
`_share_determinism`.

That is worse than the bug it replaced. An empty run labelled COMPLETE at
least looks broken; this one looks like a finding about the target.

The fix is not another guard. Restored outcomes recover their text from the
blob store -- every observation already addresses the bytes it scored, which
is what `blob_ids` is for -- so there is no longer a path where cross-run
scorers see empty strings.
"""

from __future__ import annotations

import collections
import contextlib
import os
from pathlib import Path

import pytest
from tests.conftest import make_app, make_client

import sweepeval.http.client as http_client
from sweepeval.execute.sweep import SweepStatus, asweep_target

DETERMINISM = (
    "target_determinism_at_temp0",
    "config_repeatability",
    "semantic_stability",
)


async def _sweep(root: Path, *, resume: str | None = None, stop_after: int = 0):
    """Run a sweep, optionally raising KeyboardInterrupt mid-config.

    A genuine interrupt at the transport, not a synthesised half-written run
    directory: the bug lives in what the checkpoint file and the log disagree
    about, and a hand-built fixture is free to get that agreement wrong in a
    way the real crash does not.
    """
    app = make_app("openai_clean")
    client = make_client(app)
    sent = {"n": 0}
    real = http_client.TransportClient.call

    async def counted(self, *args, **kwargs):
        sent["n"] += 1
        if stop_after and sent["n"] > stop_after:
            raise KeyboardInterrupt
        return await real(self, *args, **kwargs)

    http_client.TransportClient.call = counted  # type: ignore[method-assign]
    try:
        return await asweep_target(
            "https://mock.test" + app.scenario.paths[0],
            key="test-key-abcdefgh", client=client, root=str(root), runs=2,
            profile="quick", authorized=True, authorization_prompt=False,
            seed=7, config_cap=2, resume_run_id=resume,
        )
    finally:
        http_client.TransportClient.call = real  # type: ignore[method-assign]
        await client.aclose()


@pytest.fixture(scope="module")
def uninterrupted(tmp_path_factory):
    """What the numbers are supposed to be. Shared: slow."""
    import asyncio

    return asyncio.run(_sweep(tmp_path_factory.mktemp("ref")))


def _crash_and_resume(root: Path, cut: int):
    import asyncio

    async def go():
        with contextlib.suppress(KeyboardInterrupt):
            await _sweep(root, stop_after=cut)
        run_id = sorted(os.listdir(root / "runs"))[0]
        return await _sweep(root, resume=run_id)

    return asyncio.run(go())


# --- the bug --------------------------------------------------------------


@pytest.mark.parametrize("cut", [100, 210])
def test_a_partial_resume_reports_the_same_numbers_as_no_crash_at_all(
    uninterrupted, tmp_path_factory, cut: int
) -> None:
    """The cut points are not arbitrary. Where the interrupt lands decides how
    many runs of how many units come back restored, and most landings do not
    expose this at all: with the fix reverted, 130 and 150 -- the first two I
    tried -- pass, while 100 and 210 fail. A parametrisation that cannot fail
    under mutation is the exact defect this file was written about.

    100 corrupts all three determinism metrics on cfg-00 and propagates one to
    cfg-01 through `_share_determinism`; 210 corrupts `semantic_stability` on
    cfg-01 alone."""
    resumed = _crash_and_resume(tmp_path_factory.mktemp(f"cut{cut}"), cut)
    assert resumed.status is SweepStatus.COMPLETE

    expected = {
        row.config_id: {m: row.metrics[m].point for m in DETERMINISM if m in row.metrics}
        for row in uninterrupted.configs
    }
    for row in resumed.configs:
        for metric, want in expected.get(row.config_id, {}).items():
            assert row.metrics[metric].point == pytest.approx(want), (
                f"{row.config_id}.{metric}: crash-free {want}, resumed "
                f"{row.metrics[metric].point}"
            )


def test_a_partial_resume_does_not_write_the_rows_twice(tmp_path_factory) -> None:
    """The append-only log ended up holding both the correct set and the wrong
    one, so an offline re-score saw a config's determinism twice with two
    different answers."""
    resumed = _crash_and_resume(tmp_path_factory.mktemp("dup"), 100)
    assert resumed.store is not None

    counts = collections.Counter(
        (o.config_id, o.metric)
        for o in resumed.store.observations.read()
        if o.metric in DETERMINISM
    )
    units = sum(1 for u in resumed.units if u.family == "determinism")
    for key, count in counts.items():
        assert count == units, f"{key}: {count} rows for {units} determinism units"


def test_the_restored_text_comes_back_from_the_blob_store(tmp_path_factory) -> None:
    """The mechanism. Without it the fix is another guard, and the next
    uncovered path scores on empty strings again."""
    from sweepeval.execute.runner import _restore_text

    resumed = _crash_and_resume(tmp_path_factory.mktemp("blob"), 100)
    assert resumed.store is not None
    rows = [
        o for o in resumed.store.observations.read()
        if o.metric == "security_pass_rate" and o.blob_ids
    ]
    assert rows, "no observation addressed the bytes it scored"
    assert _restore_text(resumed.store, rows[:1]), "blob_ids pointed at nothing"


def test_a_run_whose_bodies_were_never_stored_keeps_the_stored_verdicts(
    tmp_path_factory,
) -> None:
    """`--no-store-bodies` leaves nothing to recover. Recomputing would then
    score empty strings -- the original bug -- so the previous run's cross-run
    rows stand instead, and they were computed from the same runs."""
    import asyncio

    async def go():
        root = tmp_path_factory.mktemp("nobodies")
        app = make_app("openai_clean")
        client = make_client(app)
        try:
            first = await asweep_target(
                "https://mock.test" + app.scenario.paths[0],
                key="test-key-abcdefgh", client=client, root=str(root), runs=2,
                profile="quick", authorized=True, authorization_prompt=False,
                seed=7, config_cap=1, store_text=False,
            )
            again = await asweep_target(
                "https://mock.test" + app.scenario.paths[0],
                key="test-key-abcdefgh", client=client, root=str(root), runs=2,
                profile="quick", authorized=True, authorization_prompt=False,
                seed=7, config_cap=1, store_text=False,
                resume_run_id=first.run_id,
            )
        finally:
            await client.aclose()
        return first, again

    first, again = asyncio.run(go())
    for a, b in zip(first.configs, again.configs, strict=True):
        for metric in DETERMINISM:
            if metric in a.metrics:
                assert b.metrics[metric].point == pytest.approx(a.metrics[metric].point)
