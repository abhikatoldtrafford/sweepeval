"""`extraction.ok` must describe the extraction that happened (§6.2, §5.1).

It described the opposite. `TransportClient` writes
`ExtractionPart(path=None, ok=False)` on every row it builds, which is honest
where it stands -- the declared path is chosen a layer above it, and for a
non-streamed response the transport never reads the body. Extraction then
happens in the runner, and nothing wrote the answer back. So every stored call
in every run claimed its text could not be read, while the scorers were busy
scoring that same text:

    $ jq -r '.extraction | "\\(.ok) \\(.path)"' calls.jsonl | sort | uniq -c
       3927 false null

Not a cosmetic gap. §5.1 sells the store as re-scorable offline, and the
obvious query for "which responses failed to parse" -- `select(.extraction.ok
== false)` -- selected the entire run. The field pointed a reader at 3,927
false positives and away from the handful of genuinely unreadable bodies.

The judge path had it too, which is the usual shape here: one concept, two
call sites, and a fix that lands on one. See `tests/e2e/test_judge_*` for the
other half.
"""

from __future__ import annotations

import tempfile

from tests.conftest import make_app, make_client

from sweepeval.schema.call import Call
from sweepeval.schema.hashing import sha256_hex

# --- the record itself ------------------------------------------------------


def test_a_successful_extraction_records_the_text_it_found() -> None:
    call = Call.example().with_extraction("$.choices[0].message.content", "Paris.")
    assert call.extraction.ok is True
    assert call.extraction.path == "$.choices[0].message.content"
    assert call.extraction.text_len == 6
    assert call.extraction.text_sha256 == sha256_hex(b"Paris.")


def test_an_empty_extraction_still_records_the_path_it_tried() -> None:
    """`ok=False, path=None` conflated "read nothing" with "was never asked to
    read". The path is the diagnostic -- it is what a user corrects."""
    call = Call.example().with_extraction("$.nope", "")
    assert call.extraction.ok is False
    assert call.extraction.path == "$.nope"
    assert call.extraction.text_sha256 is None
    assert call.extraction.text_len == 0


def test_the_hash_is_of_the_text_not_the_body() -> None:
    """Otherwise it duplicates `response.body_sha256` and cannot be used to
    check that a re-score read the same text out of a kept body."""
    call = Call.example().with_extraction("$.x", "hello")
    assert call.extraction.text_sha256 != call.response.body_sha256
    assert call.extraction.text_sha256 == sha256_hex(b"hello")


def test_nothing_else_on_the_row_moves() -> None:
    before = Call.example()
    after = before.with_extraction("$.x", "hello")
    assert after.model_dump(exclude={"extraction"}) == before.model_dump(
        exclude={"extraction"}
    )


# --- through a real run, which is where it was wrong ------------------------


async def _run(scenario: str):
    import sweepeval.execute.evaluate as ev

    app = make_app(scenario)
    client = make_client(app)
    try:
        return await ev.aevaluate_target(
            "https://mock.test" + app.scenario.paths[0],
            key="test-key-abcdefgh", client=client, root=tempfile.mkdtemp(), runs=1,
            authorized=True, authorization_prompt=False, seed=7,
        )
    finally:
        await client.aclose()


async def test_stored_target_rows_say_the_text_was_read() -> None:
    result = await _run("openai_clean")
    rows = [c for c in result.store.calls.read() if c.role == "target"]
    assert rows, "no target calls stored"

    read = [c for c in rows if c.extraction.ok]
    assert read, "every stored row still claims its response was unreadable"

    for call in read:
        assert call.extraction.path, "ok with no path is not a provenance record"
        assert call.extraction.text_len
        assert call.extraction.text_sha256


async def test_the_recorded_path_is_the_one_discovery_declared() -> None:
    """A hardcoded `ok=True` would pass the test above. This ties the row to
    the extraction that actually ran."""
    import sweepeval.execute.evaluate as ev

    seen: list[str | None] = []
    real = ev.discover_target

    async def patched(*args, **kwargs):
        outcome = await real(*args, **kwargs)
        seen.append(outcome.extraction.path)
        return outcome

    ev.discover_target = patched  # type: ignore[assignment]
    try:
        result = await _run("openai_clean")
    finally:
        ev.discover_target = real  # type: ignore[assignment]

    assert seen and seen[0], "discovery declared no path; the test proves nothing"
    declared = seen[0]
    paths = {
        c.extraction.path
        for c in result.store.calls.read()
        if c.role == "target" and c.extraction.ok
    }
    assert paths == {declared}, paths


async def test_a_row_whose_response_was_unreadable_still_says_so() -> None:
    """The field has to keep discriminating, or marking everything true is no
    better than marking everything false."""
    import sweepeval.execute.evaluate as ev

    app = make_app("openai_clean")
    client = make_client(app)
    real = ev.discover_target

    async def patched(*args, **kwargs):
        outcome = await real(*args, **kwargs)
        outcome.extraction.path = "$.choices[0].message.NOPE"
        return outcome

    ev.discover_target = patched  # type: ignore[assignment]
    try:
        result = await ev.aevaluate_target(
            "https://mock.test" + app.scenario.paths[0],
            key="test-key-abcdefgh", client=client, root=tempfile.mkdtemp(), runs=1,
            authorized=True, authorization_prompt=False, seed=7,
        )
    finally:
        ev.discover_target = real  # type: ignore[assignment]
        await client.aclose()

    rows = [c for c in result.store.calls.read() if c.role == "target"]
    assert rows
    assert not any(c.extraction.ok for c in rows), (
        "an unreadable run reports readable rows"
    )
    assert all(c.extraction.path == "$.choices[0].message.NOPE" for c in rows), (
        "the path that failed is what the user needs to see"
    )
