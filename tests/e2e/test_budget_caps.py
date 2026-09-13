"""A budget cap has to be a ceiling (spec §12.3).

An audit measured `BudgetCap(100, "requests")` letting 157 requests through
and `BudgetCap(200)` letting 277 through, because the counter was seeded from
per-config state only -- discovery and capability detection, which go first
and are billable, were not counted. The token cap was never checked against a
real count, and the dollar cap returned False unconditionally: a cap of one
cent completed a 757-request run, with a unit test enshrining that as intended.

Requests are counted at the transport here, not taken from the tool's own
bookkeeping, because the tool's bookkeeping is the thing under test.
"""

from __future__ import annotations

import httpx
import pytest
from tests.conftest import make_app

from sweepeval.corpus.loader import load_corpus
from sweepeval.execute.budget import BudgetCap, estimate_run
from sweepeval.execute.cost import Pricing
from sweepeval.execute.sweep import SweepStatus, asweep_target


class _Counting(httpx.AsyncBaseTransport):
    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self.inner = inner
        self.count = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.count += 1
        return await self.inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self.inner.aclose()


async def _sweep(tmp_path, cap: BudgetCap, pricing: Pricing | None = None):
    app = make_app("openai_clean")
    counting = _Counting(httpx.ASGITransport(app=app))
    client = httpx.AsyncClient(transport=counting, base_url="https://mock.test")
    try:
        result = await asweep_target(
            "https://mock.test/v1/chat/completions",
            key="test-key-abcdefgh", client=client, root=str(tmp_path), runs=2,
            profile="quick", authorized=True, authorization_prompt=False,
            seed=7, config_cap=4, cap=cap, pricing=pricing,
        )
    finally:
        await client.aclose()
    return result, counting.count


async def test_a_request_cap_is_never_exceeded(tmp_path) -> None:
    """It used to trigger only after a configuration had already run, so the
    overshoot was a whole config's worth."""
    cap = 400
    result, sent = await _sweep(tmp_path, BudgetCap(value=cap, unit="requests"))
    assert sent <= cap, f"{sent} requests against a cap of {cap}"
    assert result.status is SweepStatus.INCOMPLETE
    assert result.configs, "the cap should still buy some configurations"
    assert result.not_run


async def test_the_cap_counts_discovery_and_capability_detection(tmp_path) -> None:
    """Both are billable and both go first."""
    cap = 400
    _result, sent = await _sweep(tmp_path, BudgetCap(value=cap, unit="requests"))
    corpus = load_corpus("quick")
    scoring_only = corpus.calls_per_run * 2 * 4
    assert sent < scoring_only, (
        "the cap let through as much as an uncapped scoring phase, so the "
        "earlier phases were probably not counted"
    )


async def test_a_token_cap_buys_a_partial_sweep(tmp_path) -> None:
    """This asserted DECLINED at 30,000 tokens, which encoded the bug as the
    intent. There was NO cap value that produced a partial sweep: capability
    detection reserved a flat 200,000 tokens -- 3,333 per request, against 58
    for the entire scoring phase -- so `forbids_starting` demanded 219,000
    while a whole sweep measured about 11,000. Below the threshold it declined
    outright; above it, the cap never bound."""
    result, sent = await _sweep(tmp_path, BudgetCap(value=30_000, unit="tokens"))
    assert result.status is SweepStatus.INCOMPLETE
    assert result.configs, "the cap should still buy some configurations"
    assert result.not_run
    assert sent > 0


async def test_a_token_cap_below_one_configuration_still_declines(tmp_path) -> None:
    """The floor has to remain a floor."""
    result, sent = await _sweep(tmp_path, BudgetCap(value=20_000, unit="tokens"))
    assert result.status is SweepStatus.DECLINED
    assert sent == 0


@pytest.mark.parametrize("cap", [25_000, 32_000])
async def test_the_token_cap_has_no_dead_zone(tmp_path, cap: int) -> None:
    """Two values either side of one configuration's cost, both of which have
    to land on a real partial sweep rather than on all-or-nothing."""
    result, _sent = await _sweep(
        tmp_path / str(cap), BudgetCap(value=cap, unit="tokens")
    )
    assert result.status is SweepStatus.INCOMPLETE
    assert 0 < len(result.configs) < len(result.plan.configs)


async def test_a_dollar_cap_binds_when_pricing_is_supplied(tmp_path) -> None:
    """It returned False unconditionally, so a one-cent cap completed a
    757-request run."""
    pricing = Pricing(input_per_mtok=1.0, output_per_mtok=2.0, source="test")
    result, sent = await _sweep(
        tmp_path, BudgetCap(value=0.00005, unit="dollars"), pricing
    )
    assert result.status is SweepStatus.DECLINED
    assert not result.configs
    assert sent == 0, "a cap that cannot buy one config must send nothing"


async def test_a_dollar_cap_buys_a_partial_sweep(tmp_path) -> None:
    pricing = Pricing(input_per_mtok=1.0, output_per_mtok=10.0, source="test")
    result, _sent = await _sweep(
        tmp_path, BudgetCap(value=0.05, unit="dollars"), pricing
    )
    assert result.status is SweepStatus.INCOMPLETE
    assert 0 < len(result.configs) < len(result.plan.configs)


async def test_input_tokens_are_not_billed_at_the_output_rate(tmp_path) -> None:
    """`pricing.cost(0, tokens)` passed the combined count as the OUTPUT
    count, overstating projected spend by the price ratio -- 10x here -- in
    the direction that stops a run with budget left. The two rates differ by
    10x, so a cap set between the true and the overstated figure separates
    them."""
    from sweepeval.execute.sweep import _cap_reached

    pricing = Pricing(input_per_mtok=1.0, output_per_mtok=10.0, source="test")
    cap = BudgetCap(value=0.005, unit="dollars")

    # 1M input tokens, none out: $1.00 at the input rate, $10.00 at output.
    assert pricing.cost(1_000_000, 0) == pytest.approx(1.0)
    assert pricing.cost(0, 1_000_000) == pytest.approx(10.0)

    # 4,000 input tokens is $0.004 -- under the cap. Billed as output it is
    # $0.04, over it.
    assert not _cap_reached(
        cap, 0, tokens_in=4_000, tokens_out=0, pricing=pricing
    )
    assert _cap_reached(cap, 0, tokens_in=0, tokens_out=4_000, pricing=pricing)


async def test_a_dollar_cap_without_pricing_cannot_bind(tmp_path) -> None:
    """Inventing a price to enforce a cap is what D8 forbids. The pre-flight
    says the cost objective is in tokens; the cap is inert and does not
    silently stop the run."""
    result, _sent = await _sweep(
        tmp_path, BudgetCap(value=0.00005, unit="dollars"), None
    )
    assert result.status is SweepStatus.COMPLETE


async def test_an_uncapped_run_is_unaffected(tmp_path) -> None:
    result, _sent = await _sweep(tmp_path, BudgetCap())
    assert result.status is SweepStatus.COMPLETE
    assert len(result.configs) == len(result.plan.configs)


def test_a_cap_that_cannot_buy_one_config_says_the_arithmetic() -> None:
    estimate = estimate_run(load_corpus("quick"), configs=4, runs=2, profile="quick")
    cap = BudgetCap(value=100, unit="requests")
    assert cap.forbids_starting(estimate)
    message = cap.shortfall(estimate)
    assert "discovery and capability detection" in message
    assert "one configuration" in message
    assert "100" in message


@pytest.mark.parametrize("unit", ["requests", "tokens"])
def test_room_for_one_configuration_is_required_not_just_discovery(
    unit: str,
) -> None:
    """A cap that affords discovery but not one scored config buys no sweep,
    and spending it on discovery leaves nothing to score with."""
    estimate = estimate_run(load_corpus("quick"), configs=4, runs=2, profile="quick")
    unavoidable = (
        estimate.unavoidable_requests
        if unit == "requests"
        else sum(
            p.tokens
            for p in estimate.phases
            if p.phase in ("discovery", "capabilities")
        )
    )
    assert BudgetCap(value=unavoidable + 1, unit=unit).forbids_starting(estimate)
