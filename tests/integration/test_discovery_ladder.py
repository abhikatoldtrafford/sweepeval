"""The discovery ladder against the mock (spec §8.1-§8.4). I8 load-bearing."""

from __future__ import annotations

import pytest
from tests.conftest import make_app, make_client

from sweepeval.discovery.budget import (
    INERT_PROMPT,
    DiscoveryAborted,
    DiscoveryBudget,
)
from sweepeval.discovery.ladder import climb
from sweepeval.discovery.sniff import sniff


async def _climb(name: str, key: str | None = None, budget: DiscoveryBudget | None = None):
    app = make_app(name)
    client = make_client(app)
    try:
        return await climb(client, "https://mock.test" + app.scenario.paths[0], key, budget)
    finally:
        await client.aclose()


# --- shape identification --------------------------------------------------


async def test_identifies_the_openai_shape() -> None:
    result = await _climb("openai_clean", key="test-key-abcdefgh")
    assert result.shape.name == "openai.chat_completions"
    assert result.auth.name == "bearer"


async def test_identifies_the_anthropic_shape_and_header_auth() -> None:
    result = await _climb("anthropic_streaming", key="test-key-abcdefgh")
    assert result.shape.name == "anthropic.messages"
    assert result.auth.name == "x-api-key"


async def test_identifies_the_gemini_shape_and_query_auth() -> None:
    result = await _climb("gemini_shape", key="test-key-abcdefgh")
    assert result.shape.name == "gemini.generate_content"
    assert result.auth.name == "query"
    assert result.puts_key_in_url is True


async def test_identifies_a_bespoke_shape_by_walking_the_ladder() -> None:
    """weird_shape wants {"payload": ...}, which no built-in sends directly."""
    result = await _climb("weird_shape")
    assert result.response_json["result"]["payload"]["answer"]["value"] == "OK"


# --- I8: inert prompts and hard caps ---------------------------------------


async def test_discovery_only_ever_sends_the_inert_prompt() -> None:
    """I8.

    On an agent target a read-shaped body is still a live prompt that can
    trigger tool calls, writes, or spend. This greps every body actually sent.
    """
    sent: list[bytes] = []

    app = make_app("openai_clean")
    original = app._handle

    def spy(scope, body):  # type: ignore[no-untyped-def]
        if scope["method"] == "POST":
            sent.append(body)
        return original(scope, body)

    app._handle = spy  # type: ignore[method-assign]
    client = make_client(app)
    try:
        await climb(client, "https://mock.test/v1/chat/completions", "test-key-abcdefgh")
    finally:
        await client.aclose()

    assert sent
    for body in sent:
        text = body.decode("utf-8", errors="replace")
        assert INERT_PROMPT in text, text[:200]


async def test_the_post_budget_is_hard() -> None:
    """A target that never authenticates drives the cap and aborts."""
    budget = DiscoveryBudget(max_posts=3)
    app = make_app("openai_clean")
    client = make_client(app)
    try:
        with pytest.raises(DiscoveryAborted):
            await climb(
                client, "https://mock.test/v1/chat/completions", "wrong-key-value", budget
            )
    finally:
        await client.aclose()
    assert budget.posts <= 3


async def test_abort_lists_every_shape_and_response() -> None:
    """§8.3: the diagnostic *is* the deliverable when discovery fails."""
    app = make_app("openai_clean")
    client = make_client(app)
    try:
        with pytest.raises(DiscoveryAborted) as excinfo:
            # Wrong key: every rung 401s, nothing ever succeeds.
            await climb(client, "https://mock.test/v1/chat/completions", "wrong-key-value")
    finally:
        await client.aclose()

    message = str(excinfo.value)
    assert "discovery failed" in message
    assert "openai.chat_completions" in message
    assert "401" in message
    assert "next steps" in message


async def test_abort_records_the_error_body_excerpt() -> None:
    app = make_app("openai_clean")
    client = make_client(app)
    try:
        with pytest.raises(DiscoveryAborted) as excinfo:
            await climb(client, "https://mock.test/v1/chat/completions", "wrong-key-value")
    finally:
        await client.aclose()
    assert "invalid authentication" in str(excinfo.value)


# --- stage A: the free sniff -----------------------------------------------


async def test_sniff_finds_the_model_list() -> None:
    app = make_app("openai_clean")
    client = make_client(app)
    try:
        result = await sniff(client, "https://mock.test/v1/chat/completions")
    finally:
        await client.aclose()
    assert result.models == ("gpt-mock-large", "gpt-mock-small")
    assert result.suggested_shape == "openai.chat_completions"


async def test_sniff_reads_openapi_paths() -> None:
    app = make_app("openai_clean")
    client = make_client(app)
    try:
        result = await sniff(client, "https://mock.test/v1/chat/completions")
    finally:
        await client.aclose()
    assert "/v1/chat/completions" in result.openapi_paths


async def test_sniff_is_uninformative_on_a_bare_target() -> None:
    app = make_app("weird_shape")
    client = make_client(app)
    try:
        result = await sniff(client, "https://mock.test/invoke")
    finally:
        await client.aclose()
    assert result.models == ()
    assert result.suggested_shape is None


async def test_a_sniff_hit_collapses_the_ladder_to_one_post() -> None:
    """§8.1: a /v1/models hit should mean one confirming request, not six."""
    result = await _climb("openai_clean", key="test-key-abcdefgh")
    posts = [a for a in result.attempts] if result.attempts else []
    # The result carries no attempts list itself; assert via a fresh budget.
    budget = DiscoveryBudget()
    app = make_app("openai_clean")
    client = make_client(app)
    try:
        await climb(client, "https://mock.test/v1/chat/completions",
                    "test-key-abcdefgh", budget)
    finally:
        await client.aclose()
    assert budget.posts == 1, [a.render() for a in budget.attempts]
    assert posts == []


# --- stage C: path completion ----------------------------------------------


async def test_a_base_url_is_completed_to_the_endpoint_path() -> None:
    """§8.1 stage C, gated on every shape returning 404/405."""
    app = make_app("base_url_404s")
    client = make_client(app)
    try:
        result = await climb(client, "https://mock.test/", None)
    finally:
        await client.aclose()
    assert result.stage == "C"
    assert result.path.endswith("/v1/chat/completions")


async def test_path_completion_does_not_fire_when_the_url_answers() -> None:
    budget = DiscoveryBudget()
    app = make_app("weird_shape")
    client = make_client(app)
    try:
        result = await climb(client, "https://mock.test/invoke", None, budget)
    finally:
        await client.aclose()
    assert result.stage == "B"


# --- auth ladder -----------------------------------------------------------


async def test_the_auth_ladder_stops_at_the_first_success() -> None:
    budget = DiscoveryBudget()
    app = make_app("anthropic_streaming")
    client = make_client(app)
    try:
        await climb(client, "https://mock.test/v1/messages", "test-key-abcdefgh", budget)
    finally:
        await client.aclose()
    # bearer is tried first and 401s, x-api-key succeeds: two attempts, not four.
    auths = [a.note for a in budget.attempts]
    assert auths[-1] == "auth=x-api-key"
    assert sum(1 for a in auths if a == "auth=query") == 0


async def test_no_key_means_a_single_unauthenticated_attempt() -> None:
    budget = DiscoveryBudget()
    app = make_app("weird_shape")
    client = make_client(app)
    try:
        await climb(client, "https://mock.test/invoke", None, budget)
    finally:
        await client.aclose()
    assert all(a.note == "auth=none" for a in budget.attempts)
