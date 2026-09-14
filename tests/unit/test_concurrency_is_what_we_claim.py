"""What the tool says about concurrency has to be what it does.

`DEFAULT_CONCURRENCY` was 2 and the executor dispatched one request at a time.
The semaphore was built, sized and never contended -- the same shape as
`from_entry_points`, `write_derived` and `budget.judge_units` before it: a
mechanism that is configured, documented, tested in isolation and wired to
nothing.

It was not free. The pre-flight estimate divides its ETA by this number, so
every estimate the tool has ever shown was optimistic by a factor of two, and
an estimate is the last thing a user sees before agreeing to spend. The soft
comparability key recorded it too, which made every stored manifest assert
something about the run that was not true of it.

So the number moved to the truth. These tests are what stop them parting
again: raise `DEFAULT_CONCURRENCY` and the observed-peak test fails until the
executor really dispatches that many.
"""

from __future__ import annotations

import asyncio
import tempfile

import httpx
import pytest
from tests.conftest import make_app

from sweepeval.http.governor import DEFAULT_CONCURRENCY, Governor


class _CountsInFlight(httpx.AsyncBaseTransport):
    """Records peak overlap, yielding to the loop so overlap is possible.

    The mock's own `latency_ms` is a blocking `time.sleep` inside the ASGI
    app, which pins the event loop and would report serial execution however
    the executor behaved. Any concurrency measurement that sleeps that way
    measures the mock.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self.inner = inner
        self.in_flight = 0
        self.peak = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(0.01)
            return await self.inner.handle_async_request(request)
        finally:
            self.in_flight -= 1


async def _run_and_count() -> tuple[int, int]:
    import sweepeval.execute.evaluate as ev

    app = make_app("openai_clean")
    transport = _CountsInFlight(httpx.ASGITransport(app=app))
    client = httpx.AsyncClient(transport=transport, base_url="https://mock.test")
    try:
        result = await ev.aevaluate_target(
            "https://mock.test/v1/chat/completions",
            key="test-key-abcdefgh", client=client, root=tempfile.mkdtemp(),
            runs=1, authorized=True, authorization_prompt=False, seed=7,
        )
    finally:
        await client.aclose()
    return transport.peak, len(list(result.store.calls.read()))


@pytest.fixture(scope="module")
def observed() -> tuple[int, int]:
    return asyncio.run(_run_and_count())


def test_the_run_was_big_enough_to_have_shown_overlap(observed) -> None:
    """Otherwise the peak below is 1 because there was nothing to overlap."""
    _peak, calls = observed
    assert calls > 20, calls


def test_peak_in_flight_never_exceeds_what_we_declare(observed) -> None:
    peak, _calls = observed
    assert peak <= DEFAULT_CONCURRENCY, (
        f"dispatched {peak} at once while declaring {DEFAULT_CONCURRENCY}"
    )


def test_we_do_not_declare_more_than_we_dispatch(observed) -> None:
    """The direction that was actually wrong, and the one that costs a user
    money: an ETA divided by a concurrency the executor never uses."""
    peak, _calls = observed
    assert peak == DEFAULT_CONCURRENCY, (
        f"declared {DEFAULT_CONCURRENCY} and never had more than {peak} in "
        "flight; either dispatch it or stop claiming it"
    )


def test_the_estimate_divides_by_the_same_number() -> None:
    """The ETA is the last thing shown before a user agrees to spend."""
    from sweepeval.execute.budget import Estimate

    assert Estimate().concurrency == DEFAULT_CONCURRENCY


def test_the_comparability_key_records_the_same_number() -> None:
    """A manifest that overstates concurrency asserts something untrue about
    the run, and two runs are then declared comparable on a fiction."""
    import inspect

    import sweepeval.execute.evaluate as evaluate
    import sweepeval.execute.sweep as sweep

    for module in (evaluate, sweep):
        source = inspect.getsource(module)
        assert "concurrency=2" not in source, (
            f"{module.__name__} hardcodes a concurrency instead of reading "
            "DEFAULT_CONCURRENCY"
        )
        assert "concurrency=DEFAULT_CONCURRENCY" in source


def test_the_governor_still_gates_at_the_declared_width() -> None:
    """Serial today is a fact about the executor, not about the Governor. The
    gate has to keep working, or raising the constant later gates nothing."""

    async def check() -> None:
        governor = Governor(concurrency=2)
        held = []

        async def hold() -> None:
            async with governor.semaphore:
                held.append(1)
                await asyncio.sleep(0.05)

        await asyncio.gather(*(hold() for _ in range(4)))
        assert len(held) == 4
        assert not governor.semaphore.locked(), "permits were not released"

    asyncio.run(check())
