"""The transport client against the mock (spec §7, D29)."""

from __future__ import annotations

from tests.conftest import make_app, make_client

from sweepeval.http.client import TransportClient, supports_usage_in_stream
from sweepeval.http.governor import Budget, Governor
from sweepeval.schema.call import ErrorClass


def _transport(app: object, **kwargs: object) -> TransportClient:
    governor = kwargs.pop("governor", None) or Governor(seed=1)
    return TransportClient(
        make_client(app),  # type: ignore[arg-type]
        governor,  # type: ignore[arg-type]
        run_id="r1",
        shape=str(kwargs.pop("shape", "openai.chat_completions")),
    )


async def _one(name: str, body: dict[str, object], **kwargs: object) -> list[object]:
    app = make_app(name)
    client = make_client(app)
    transport = TransportClient(
        client,
        kwargs.pop("governor", None) or Governor(seed=1),  # type: ignore[arg-type]
        run_id="r1",
        shape="openai.chat_completions",
    )
    try:
        return await transport.call(
            app.scenario.paths[0], body, config_id="c1", unit_id="u1", **kwargs  # type: ignore[arg-type]
        )
    finally:
        await client.aclose()


# --- one row per attempt --------------------------------------------------


async def test_a_successful_call_produces_exactly_one_row() -> None:
    results = await _one("echoes_the_prompt", {"messages": [{"role": "user", "content": "hi"}]})
    assert len(results) == 1
    assert results[0].call.attempt == 1  # type: ignore[attr-defined]
    assert results[0].call.response.error_class is ErrorClass.ok  # type: ignore[attr-defined]


async def test_a_terminal_error_is_not_retried() -> None:
    """One row, not four: a 400 will still be a 400."""
    results = await _one("openai_clean", {"messages": []})  # no auth header -> 401
    assert len(results) == 1
    assert results[0].call.response.error_class is ErrorClass.terminal  # type: ignore[attr-defined]


async def test_a_retryable_error_produces_one_row_per_attempt() -> None:
    """A run that succeeded only after three retries must look different from
    one that succeeded immediately (§6.4)."""
    app = make_app("ratelimit_storm")
    client = make_client(app)
    governor = Governor(seed=1, base_delay_s=0.0, max_delay_s=0.0)
    transport = TransportClient(client, governor, run_id="r1", shape="openai.chat_completions")
    try:
        for _ in range(5):
            await transport.call("/v1/chat/completions", {"messages": []},
                                 config_id="c1", unit_id="u1")
        results = await transport.call("/v1/chat/completions", {"messages": []},
                                       config_id="c1", unit_id="u1")
    finally:
        await client.aclose()

    assert len(results) == governor.max_attempts
    assert [r.call.attempt for r in results] == [1, 2, 3, 4]
    assert all(r.call.response.error_class is ErrorClass.retryable for r in results)


async def test_retries_are_excluded_from_the_latency_population() -> None:
    app = make_app("ratelimit_storm")
    client = make_client(app)
    transport = TransportClient(
        client, Governor(seed=1, base_delay_s=0.0, max_delay_s=0.0),
        run_id="r1", shape="openai.chat_completions",
    )
    try:
        for _ in range(6):
            await transport.call("/v1/chat/completions", {"messages": []},
                                 config_id="c1", unit_id="u1")
        results = await transport.call("/v1/chat/completions", {"messages": []},
                                       config_id="c1", unit_id="u1")
    finally:
        await client.aclose()
    assert not any(r.call.counts_toward_latency for r in results)


# --- timing ---------------------------------------------------------------


async def test_total_excludes_queue_time() -> None:
    """§6.4: latency must measure the target, not the harness."""
    results = await _one("openai_clean", {"messages": []},
                         headers={"authorization": "Bearer test-key-abcdefgh"})
    timing = results[0].call.timing  # type: ignore[attr-defined]
    assert timing.total_ms >= 0
    assert timing.queue_ms >= 0
    assert timing.total_ms < timing.total_ms + timing.queue_ms + 1


async def test_ttft_is_recorded() -> None:
    results = await _one("openai_clean", {"messages": []},
                         headers={"authorization": "Bearer test-key-abcdefgh"})
    assert results[0].call.timing.ttft_ms is not None  # type: ignore[attr-defined]


# --- tokens ---------------------------------------------------------------


async def test_a_usage_block_is_recorded_as_measured() -> None:
    results = await _one("openai_clean", {"messages": []},
                         headers={"authorization": "Bearer test-key-abcdefgh"})
    tokens = results[0].call.tokens  # type: ignore[attr-defined]
    assert tokens.source == "MEASURED"
    assert tokens.in_ is not None


async def test_a_missing_usage_block_falls_back_to_estimated() -> None:
    """§12.4 tier 2. The label is what keeps the cost objective honest."""
    results = await _one("no_usage_block", {"messages": []})
    assert results[0].call.tokens.source == "ESTIMATED"  # type: ignore[attr-defined]


def test_usage_in_stream_is_requested_only_where_the_shape_supports_it() -> None:
    assert supports_usage_in_stream("openai.chat_completions")
    assert not supports_usage_in_stream("anthropic.messages")


# --- governor integration -------------------------------------------------


async def test_spend_is_recorded_against_the_budget() -> None:
    app = make_app("echoes_the_prompt")
    client = make_client(app)
    governor = Governor(seed=1)
    transport = TransportClient(client, governor, run_id="r1", shape="openai.chat_completions")
    try:
        await transport.call("/v1/chat/completions", {"messages": []},
                             config_id="c1", unit_id="u1")
    finally:
        await client.aclose()
    assert governor.budget.requests == 1


async def test_the_budget_cap_stops_the_call() -> None:
    import pytest

    from sweepeval.http.governor import BudgetExceeded

    app = make_app("echoes_the_prompt")
    client = make_client(app)
    governor = Governor(seed=1, budget=Budget(max_requests=1))
    transport = TransportClient(client, governor, run_id="r1", shape="openai.chat_completions")
    try:
        await transport.call("/v1/chat/completions", {"messages": []},
                             config_id="c1", unit_id="u1")
        with pytest.raises(BudgetExceeded):
            await transport.call("/v1/chat/completions", {"messages": []},
                                 config_id="c1", unit_id="u1")
    finally:
        await client.aclose()


async def test_five_consecutive_terminal_errors_trip_the_breaker() -> None:
    app = make_app("openai_clean")
    client = make_client(app)
    governor = Governor(seed=1)
    transport = TransportClient(client, governor, run_id="r1", shape="openai.chat_completions")
    try:
        for _ in range(5):
            await transport.call("/v1/chat/completions", {"messages": []},
                                 config_id="c1", unit_id="u1")
    finally:
        await client.aclose()
    assert governor.is_tripped("c1")
