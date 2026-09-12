"""No secret reaches disk (spec §6.6, D38).

Exhaustive rather than a spot check, because ``Store`` is the only way to
obtain a writer: there is no other constructor path a caller could use to reach
past the redactor.

Run against ``query_param_auth``, the scenario where the key lands in the URL —
the fourth rung of §8.4's auth ladder, and the case that matters most because
``baseline.json`` is designed to be committed to a user's repository.
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import make_app, make_client

from sweepeval.http.client import TransportClient
from sweepeval.http.governor import Governor
from sweepeval.store.redaction import Redactor
from sweepeval.store.run import Store, new_run_id

KEY = "sk-supersecretkeyvalue"


def _all_bytes_under(root: Path) -> str:
    return "".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in root.rglob("*")
        if path.is_file()
    )


async def test_a_query_param_key_never_reaches_any_artifact(tmp_path: Path) -> None:
    app = make_app("query_param_auth")
    client = make_client(app)
    store = Store(tmp_path, new_run_id(), Redactor(secrets=[KEY]))
    transport = TransportClient(
        client, Governor(seed=1), run_id=store.run_id, shape="openai.chat_completions"
    )

    try:
        results = await transport.call(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hello"}]},
            config_id="c1",
            unit_id="u1",
            params={"api_key": KEY},
        )
    finally:
        await client.aclose()

    # Write everything a real run would write.
    for result in results:
        store.calls.append(result.call)
        store.blobs.put_text(result.text or "empty", always=True)
        store.blobs.put_text(
            f"request to https://mock.test/v1/chat/completions?api_key={KEY}",
            always=True,
        )
    store.state_for("c1").mark_complete("c1", "u1", 0)

    assert KEY not in _all_bytes_under(tmp_path)


async def test_the_call_succeeded_so_the_test_is_not_vacuous(tmp_path: Path) -> None:
    """A 401 would leave nothing to redact and the test above would pass for
    the wrong reason."""
    app = make_app("query_param_auth")
    client = make_client(app)
    transport = TransportClient(
        client, Governor(seed=1), run_id="r1", shape="openai.chat_completions"
    )
    try:
        results = await transport.call(
            "/v1/chat/completions",
            {"messages": []},
            config_id="c1",
            unit_id="u1",
            params={"api_key": KEY},
        )
    finally:
        await client.aclose()
    assert results[0].call.response.status == 200


def test_the_scanner_can_actually_find_content(tmp_path: Path) -> None:
    """Mutation guard: prove the assertion is capable of failing.

    Without this, a bug that stopped anything being written would make the
    no-secrets tests pass trivially — the failure mode this whole file exists
    to rule out.

    Note the marker is deliberately *not* provider-shaped. An ``sk-`` prefixed
    string is masked by the token patterns even when it was never supplied as a
    secret, so it could not demonstrate that the scanner works.
    """
    marker = "PLAIN-MARKER-NOT-A-CREDENTIAL"
    store = Store(tmp_path, new_run_id(), Redactor(secrets=[]))
    store.blobs.put_text(f"contains {marker}", always=True)
    assert marker in _all_bytes_under(tmp_path)


def test_provider_shaped_tokens_are_masked_even_when_never_supplied(
    tmp_path: Path,
) -> None:
    """Defence in depth (§6.6).

    A target's own error body can echo a credential back at us that the user
    never passed on the command line, so the token patterns apply regardless of
    what was declared as a secret.
    """
    store = Store(tmp_path, new_run_id(), Redactor(secrets=[]))
    store.blobs.put_text(f"error echoed {KEY} back", always=True)
    assert KEY not in _all_bytes_under(tmp_path)
