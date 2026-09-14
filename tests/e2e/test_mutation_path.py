"""Error-guided mutation, exercised end to end (spec §8.2).

Until this file, nothing in CI reached it. Every scenario answered a bare
``{"messages": [...]}``, so the ladder found a 200 on its first shape and
stopped; the mutation code was covered only by unit tests over canned error
strings, which is to say the *parsing* was tested and the *walk* was not.

Pointing discovery at api.openai.com on 2026-09-12 found three bugs living in
that gap, none of which the suite could see. `demands_a_model` is built to be
the endpoint that finds them: it 400s on a bare body with OpenAI's own wording
-- prose, unquoted, ``error.param`` null, the field name *before* the keyword
-- and it rejects a ``model`` it does not serve, so a mutation that invents a
plausible string instead of reading one off ``/v1/models`` still fails.

Running it turned up a fourth. The enumeration ranked `rename` above `add`,
so the walk's first move against "you must provide a model parameter" was to
move the prompt into the model field -- satisfying the complaint by destroying
the request. The endpoint then errored about the field it had just lost, and
those mutations came out of the same attempt pool, one pool for the whole tree.
`add:model`, which fixes this endpoint in a single request, was generated,
ranked second, and never sent. Discovery aborted with "request budget
exhausted" against an endpoint it was one request from reading.

Adding is non-destructive and renaming is not, so `add` goes first now. The
scenarios that are only found by renaming are the regression risk, and two of
them are asserted below.
"""

from __future__ import annotations

import json

import pytest
from tests.conftest import make_app, make_client

from sweepeval.discovery.runner import discover_target
from sweepeval.mock.app import MockApp

KEY = "test-key-abcdefgh"
URL = "https://mock.test/v1/chat/completions"


async def _discover(scenario: str = "demands_a_model"):
    app = make_app(scenario)
    client = make_client(app)
    try:
        return await discover_target(client, URL, key=KEY)
    finally:
        await client.aclose()


# --- the scenario has to actually be hostile --------------------------------


async def test_a_bare_body_is_rejected_the_way_openai_rejects_it() -> None:
    """Without this the whole file passes on an endpoint that never mutates --
    which is precisely how the judge tests passed with the judge never called.

    The wording is the payload. A message with quotes, or with "missing", or
    with the name after the keyword, is one every pattern already caught.
    """
    app = make_app("demands_a_model")
    client = make_client(app)
    try:
        response = await client.post(
            URL,
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {KEY}"},
        )
    finally:
        await client.aclose()

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["message"] == "you must provide a model parameter"
    assert error["param"] is None, "a populated param is the easy case"
    assert "'" not in error["message"] and '"' not in error["message"]


async def test_a_model_the_endpoint_does_not_serve_is_rejected() -> None:
    """The mutator once guessed ``model="default"``. No provider serves that,
    and an endpoint that accepted anything would have let the guess pass."""
    app = make_app("demands_a_model")
    client = make_client(app)
    try:
        response = await client.post(
            URL,
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "model": "default",
            },
            headers={"Authorization": f"Bearer {KEY}"},
        )
    finally:
        await client.aclose()
    assert response.status_code == 400
    assert "does not exist" in response.json()["error"]["message"]


# --- and discovery has to get through it ------------------------------------


@pytest.fixture(scope="module")
def discovered():
    import asyncio

    return asyncio.run(_discover())


def test_discovery_reads_the_endpoint_it_had_to_mutate_into(discovered) -> None:
    assert discovered.ladder.shape.name == "openai.chat_completions"
    assert discovered.extraction.path == "$.choices[0].message.content"


def test_the_field_name_came_out_of_prose_and_was_added_not_renamed(
    discovered,
) -> None:
    """``rename:messages->model`` satisfies the complaint by destroying the
    field carrying the prompt, and it used to be enumerated first. The
    surviving body still has to carry the prompt."""
    assert discovered.ladder.mutations == ("add:model",), (
        discovered.ladder.mutations
    )
    assert "messages" in discovered.ladder.body


def test_the_model_was_read_off_the_endpoint_not_invented(discovered) -> None:
    """Which also proves the metadata sniff authenticated: ``/v1/models`` is
    behind the same bearer token, and a sniff that skipped it -- as one did --
    leaves the mutator with nothing but a guess."""
    served = {"gpt-mock-large", "gpt-mock-small"}
    assert discovered.ladder.body["model"] in served


def test_it_did_not_get_there_by_spending_the_whole_budget(discovered) -> None:
    """The failure mode this replaced was an abort at exactly 25 posts. A fix
    that merely raised the cap would pass every assertion above."""
    assert discovered.budget.posts <= 12, discovered.budget.posts


def test_no_second_level_mutation_was_needed(discovered) -> None:
    """One request, not a descent. Every attempt sent carries at most one
    mutation label, because the fix was available at depth 1 and is now the
    first thing tried."""
    depths = [
        len(attempt.mutation.split(",")) if attempt.mutation else 0
        for attempt in discovered.budget.attempts
    ]
    assert max(depths) <= 1


# --- the endpoints that need renaming must still be discovered --------------


@pytest.mark.parametrize(
    ("scenario", "key"), [("weird_shape", None), ("gemini_shape", KEY)]
)
async def test_shapes_that_are_found_by_renaming_still_are(
    scenario: str, key: str | None
) -> None:
    """Reordering changed which mutation is tried when, so the scenarios that
    depend on ``rename`` are the regression risk, not the new one."""
    app = make_app(scenario)
    client = make_client(app)
    try:
        outcome = await discover_target(
            client, "https://mock.test" + app.scenario.paths[0], key=key
        )
    finally:
        await client.aclose()
    assert outcome.extraction.path


async def test_two_missing_fields_still_reach_depth_two_within_the_caps() -> None:
    """Reordering must not have cost the second level.

    An endpoint that withholds one field per error needs a mutation applied to
    the result of a mutation. This is also where the caps have to hold: the
    depth limit and the one shared attempt pool are what keep a "guided
    search" from turning into hundreds of requests against a stranger.
    """
    from sweepeval.discovery.mutate import MAX_MUTATION_ATTEMPTS, MAX_MUTATION_DEPTH

    scenario = make_app("demands_a_model").scenario.model_copy(
        update={"require_fields": ("model", "tenant_id")}
    )
    app = MockApp(scenario)
    client = make_client(app)
    try:
        outcome = await discover_target(client, URL, key=KEY)
    finally:
        await client.aclose()

    assert outcome.ladder.mutations == ("add:model", "add:tenant_id")
    assert outcome.extraction.path

    labelled = [a for a in outcome.budget.attempts if a.mutation]
    depths = [len(a.mutation.split(",")) for a in labelled]
    assert max(depths) <= MAX_MUTATION_DEPTH
    assert len(labelled) <= MAX_MUTATION_ATTEMPTS * len(outcome.budget.attempts)


def test_the_scenario_is_the_one_described(
) -> None:
    from pathlib import Path

    import yaml

    raw = yaml.safe_load(
        Path("tests/scenarios/demands_a_model.yaml").read_text(encoding="utf-8")
    )
    assert raw["require_fields"] == ["model"]
    assert raw["expose_openapi"] is False, (
        "an OpenAPI document would hand discovery the schema and skip mutation"
    )
    assert json.dumps(raw)  # serialisable, i.e. no stray python objects
