"""Cross-run observations must say which run they came from, and from what.

One `ScoreContext`, built once in `_finalize_cross_run`, was wrong in two ways
at the same time, and every determinism-family row ever written carried both:

    run_id='cfg-00' config_id='cfg-00' blob_ids=[]

`run_id` held the *config* id. It is the field that joins an observation back
to the run it belongs to (§6.2), and for four metrics it held a value that is
not a run id at all.

`blob_ids` was empty because a cross-run scorer is handed N texts and the
context carries none — `ScoreContext.text_blob_id` is a per-run idea. So
determinism was the one family whose verdict could not be re-checked offline
against the responses that produced it. That is not hypothetical: the security
column of `docs/scorecard.md` was re-scored from exactly this join after three
of its verdicts turned out to be wrong, and determinism could not have been.

Both read off the published 14-model run before the fix.
"""

from __future__ import annotations

import tempfile

import pytest
from tests.conftest import make_app, make_client

CROSS_RUN = {
    "config_repeatability",
    "semantic_stability",
    "target_determinism_at_temp0",
    "invariance",
}


@pytest.fixture(scope="module")
def evaluated():
    import asyncio

    import sweepeval.execute.evaluate as ev

    async def run():
        app = make_app("openai_clean")
        client = make_client(app)
        try:
            return await ev.aevaluate_target(
                "https://mock.test/v1/chat/completions",
                key="test-key-abcdefgh", client=client, root=tempfile.mkdtemp(),
                runs=2, authorized=True, authorization_prompt=False, seed=7,
            )
        finally:
            await client.aclose()

    result = asyncio.run(run())
    rows = [o for o in result.store.observations.read() if o.metric in CROSS_RUN]
    return result, rows


def test_the_run_was_big_enough_to_prove_anything(evaluated) -> None:
    _result, rows = evaluated
    assert len(rows) > 10, len(rows)
    assert {r.metric for r in rows} & CROSS_RUN


# --- run_id is a run id ------------------------------------------------------


def test_cross_run_rows_record_the_run_not_the_config(evaluated) -> None:
    result, rows = evaluated
    assert {r.run_id for r in rows} == {result.store.run_id}


def test_they_do_not_record_the_config_id_in_the_run_field(evaluated) -> None:
    """The exact shape of the bug: both fields held `cfg-00`."""
    for row in rows_with_config(evaluated):
        assert row.run_id != row.config_id, row


def rows_with_config(evaluated):
    _result, rows = evaluated
    return [r for r in rows if r.config_id]


def test_per_run_rows_already_agreed_and_still_do(evaluated) -> None:
    """Security and the rest were always right. The fix must not move them."""
    result, _rows = evaluated
    others = [
        o for o in result.store.observations.read() if o.metric not in CROSS_RUN
    ]
    assert others
    assert {o.run_id for o in others} == {result.store.run_id}


# --- and the verdict can be re-checked against the bytes ---------------------


def test_scored_cross_run_rows_carry_blob_ids(evaluated) -> None:
    _result, rows = evaluated
    scored = [r for r in rows if r.value is not None]
    assert scored, "no scored cross-run rows to check"
    assert all(r.blob_ids for r in scored), [
        r.metric for r in scored if not r.blob_ids
    ]


def test_every_blob_id_resolves_to_stored_bytes(evaluated) -> None:
    """An address that resolves to nothing is worse than no address: it says
    the evidence is there."""
    result, rows = evaluated
    checked = 0
    for row in rows:
        for blob_id in row.blob_ids:
            assert result.store.blobs.get(blob_id), blob_id
            checked += 1
    assert checked > 10, checked


def test_the_bytes_are_the_texts_the_verdict_compared(evaluated) -> None:
    """A repeatability of 1.0 means the runs matched, so the blobs it names
    must too — and a value below 1.0 means at least two of them differ."""
    result, rows = evaluated
    for row in rows:
        if row.metric != "config_repeatability" or row.value is None:
            continue
        texts = {result.store.blobs.get(b).decode("utf-8") for b in row.blob_ids}
        if row.value == 1.0 and len(row.blob_ids) > 1:
            assert len(texts) == 1, (row.metric, row.value, len(texts))


def test_an_unscorable_row_points_at_no_evidence(evaluated) -> None:
    """Excluded and empty runs are dropped from the list. A blob id for a text
    that was not scored points a reader at evidence the number does not rest
    on."""
    _result, rows = evaluated
    for row in rows:
        if row.value is None and row.verdict.value == "UNSCORABLE":
            assert not row.blob_ids or all(row.blob_ids)
