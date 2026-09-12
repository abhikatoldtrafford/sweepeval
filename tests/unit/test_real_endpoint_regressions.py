"""Four discovery bugs found by pointing the tool at the real OpenAI API.

Every one of them passed the whole mock suite. The mock answers a bare
`{"messages": [...]}` body, so nothing in it ever exercised an endpoint that
*demands* a field, and the entire error-guided mutation path — six enumerated
transformations, a budget, a depth limit, all of it — had never once run
against an error message a real service produces.

The four, in the order discovery hit them:

1. the field name could not be read out of OpenAI's error at all;
2. the metadata sniff did not authenticate, so `/v1/models` 401'd;
3. the mutator guessed `model="default"`, which does not exist;
4. a body reached by mutation was flattened, dropping multi-turn history and
   every swept parameter.

They are unit tests because each failure is reachable with no network, once
you know the shape of the input that causes it. Not knowing that is what cost
the discovery run.
"""

from __future__ import annotations

from typing import Any

import pytest

from sweepeval.discovery.auth import AuthMethod
from sweepeval.discovery.budget import INERT_PROMPT
from sweepeval.discovery.ladder import LadderResult, body_for_turns
from sweepeval.discovery.mutate import field_names_in_error, mutations_for
from sweepeval.discovery.shapes import ladder_order
from sweepeval.discovery.sniff import SniffResult

# Verbatim, from api.openai.com on a POST with no model.
OPENAI_NO_MODEL = (
    b'{\n  "error": {\n    "message": "you must provide a model parameter",\n'
    b'    "type": "invalid_request_error",\n    "param": null,\n'
    b'    "code": null\n  }\n}'
)


def _shape(name: str = "openai.chat_completions") -> Any:
    return next(s for s in ladder_order() if s.name == name)


def _ladder(body: dict[str, Any], mutations: tuple[str, ...]) -> LadderResult:
    return LadderResult(
        shape=_shape(),
        path="https://api.openai.com/v1/chat/completions",
        auth=AuthMethod("bearer"),
        body=body,
        response_json={},
        sniff=SniffResult(),
        stage="B",
        mutations=mutations,
    )


# --- 1. the name comes before the keyword ---------------------------------


def test_a_field_name_before_the_keyword_is_read() -> None:
    """"you must provide a model parameter" -- no quotes, no "missing", and
    error.param is null. Every pattern looked for the name *after* the
    keyword, so nothing was extracted and mutation never fired."""
    assert "model" in field_names_in_error(OPENAI_NO_MODEL)


def test_a_field_name_after_the_keyword_still_works() -> None:
    assert "model" in field_names_in_error(b'{"detail": "missing parameter model"}')


def test_grammar_words_are_not_mistaken_for_field_names() -> None:
    """"provide a model parameter" offers "a" to the before-keyword pattern as
    readily as "model"; adding `a` to a body wastes a mutation attempt out of
    six."""
    names = field_names_in_error(OPENAI_NO_MODEL)
    assert "a" not in names
    assert not {n.lower() for n in names} & {"the", "an", "a", "invalid"}


def test_the_mutator_now_fires_on_that_error() -> None:
    body = _shape().build(INERT_PROMPT)
    kinds = {m.label() for m, _ in mutations_for(body, OPENAI_NO_MODEL, INERT_PROMPT)}
    assert "add:model" in kinds


# --- 2 and 3. the added model has to be a real one ------------------------


def test_a_discovered_model_id_is_used_instead_of_a_guess() -> None:
    """"default" is not a model OpenAI has, so the guess turned "you must
    provide a model" into "the model `default` does not exist" and discovery
    failed one step later having learned nothing."""
    body = _shape().build(INERT_PROMPT)
    mutated = {
        m.label(): b
        for m, b in mutations_for(
            body, OPENAI_NO_MODEL, INERT_PROMPT, {"model": "gpt-4o-mini"}
        )
    }
    assert mutated["add:model"]["model"] == "gpt-4o-mini"


def test_without_a_hint_it_still_produces_something() -> None:
    body = _shape().build(INERT_PROMPT)
    mutated = {
        m.label(): b for m, b in mutations_for(body, OPENAI_NO_MODEL, INERT_PROMPT)
    }
    assert mutated["add:model"]["model"]


def test_the_preferred_model_skips_non_chat_and_prefers_a_cheap_one() -> None:
    from sweepeval.discovery.ladder import _preferred_model

    listed = (
        "text-embedding-3-large",
        "whisper-1",
        "gpt-4o",
        "gpt-4o-mini",
        "dall-e-3",
    )
    assert _preferred_model(listed) == "gpt-4o-mini"


def test_the_preferred_model_never_returns_nothing() -> None:
    """Even an all-non-chat list has to yield an id: an inert prompt against
    the wrong model still produces an error message worth reading."""
    from sweepeval.discovery.ladder import _preferred_model

    assert _preferred_model(("text-embedding-3-large", "whisper-1"))


async def test_the_sniff_retries_metadata_with_the_key() -> None:
    """OpenAI requires auth on /v1/models. Unauthenticated, the sniff learned
    no models and suggested no shape."""
    import httpx

    from sweepeval.discovery.sniff import sniff

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization", ""))
        if request.url.path in ("/v1/models", "/models"):
            if not request.headers.get("authorization"):
                return httpx.Response(401, json={"error": "unauthorized"})
            return httpx.Response(200, json={"data": [{"id": "gpt-4o-mini"}]})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        result = await sniff(client, "https://api.example.com/v1/chat/completions", "k")

    assert result.models == ("gpt-4o-mini",)
    assert any(h.startswith("Bearer ") for h in seen)


async def test_the_sniff_does_not_retry_a_404() -> None:
    """No credential fixes a path that does not exist, and the retry costs a
    request against a stranger's endpoint."""
    import httpx

    from sweepeval.discovery.sniff import sniff

    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await sniff(client, "https://api.example.com/v1/chat/completions", "k")

    assert len(calls) == len(set(calls)), calls


# --- 4. an additive mutation keeps the shape ------------------------------


TURNS = [("user", "one"), ("assistant", "ok"), ("user", "two")]


def test_an_additive_mutation_keeps_multi_turn_history() -> None:
    """OpenAI is discovered as `openai.chat_completions + add:model`. Treating
    that like a structural mutation collapsed every multi-turn probe into one
    user message, so the context family measured nothing."""
    ladder = _ladder(
        {**_shape().build(INERT_PROMPT), "model": "gpt-4.1-mini"}, ("add:model",)
    )
    body = body_for_turns(ladder, TURNS)
    assert len(body["messages"]) == 3
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]


def test_an_additive_mutation_still_applies_swept_parameters() -> None:
    """Temperature was dropped entirely, so the temperature axis swept nothing."""
    ladder = _ladder(
        {**_shape().build(INERT_PROMPT), "model": "gpt-4.1-mini"}, ("add:model",)
    )
    body = body_for_turns(ladder, TURNS, temperature=0.7)
    assert body["temperature"] == 0.7


def test_a_swept_model_beats_the_one_discovery_happened_to_probe_with() -> None:
    """The other merge order pins `model` to whatever the sniff picked, and
    the model axis silently does nothing."""
    ladder = _ladder(
        {**_shape().build(INERT_PROMPT), "model": "gpt-4.1-mini"}, ("add:model",)
    )
    assert body_for_turns(ladder, TURNS, model="gpt-4o-mini")["model"] == "gpt-4o-mini"
    # ...and the discovered one is still the default when nothing is swept.
    assert body_for_turns(ladder, TURNS)["model"] == "gpt-4.1-mini"


@pytest.mark.parametrize(
    ("body", "mutations"),
    [
        ({"prompt": INERT_PROMPT}, ("rename:messages->prompt",)),
        ({"input": {"messages": [{"role": "user", "content": INERT_PROMPT}]}},
         ("nest:input",)),
    ],
)
def test_a_structural_mutation_still_flattens(
    body: dict[str, Any], mutations: tuple[str, ...]
) -> None:
    """Where the shape can no longer build a body the target accepts,
    flattening is the honest fallback -- and rebuilding from the shape would
    put the text somewhere the target never reads."""
    rebuilt = body_for_turns(_ladder(body, mutations), TURNS)
    assert "messages" not in rebuilt or rebuilt.get("messages") != [
        {"role": r, "content": t} for r, t in TURNS
    ]


def test_a_rename_is_not_mistaken_for_an_addition() -> None:
    """A rename removes a key the shape built. Anything less strict than
    "every shape key still present and unchanged" would let it look additive
    and rebuild a body the target rejects."""
    from sweepeval.discovery.ladder import _additive_only

    renamed = _ladder({"prompt": INERT_PROMPT}, ("rename:messages->prompt",))
    assert _additive_only(renamed) is None

    added = _ladder(
        {**_shape().build(INERT_PROMPT), "model": "m"}, ("add:model",)
    )
    assert _additive_only(added) == {"model": "m"}
