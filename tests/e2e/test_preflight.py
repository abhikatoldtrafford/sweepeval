"""I9: no billable request precedes the estimate (spec §12.3).

The sweep path was correct. `evaluate`, `baseline` and `gate` share one helper
that had no `confirm`, no cap and no `estimate_run` call at all -- an audit
measured 133 billable requests sent before anything was shown to the user, the
first three of them before a single probe.

The counter here wraps the transport, so it counts what actually left the
process rather than what the code thinks it sent.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from tests.conftest import make_app

from sweepeval.execute.budget import BudgetCap, Estimate
from sweepeval.execute.evaluate import aevaluate_target
from sweepeval.execute.sweep import asweep_target


class _CountingTransport(httpx.AsyncBaseTransport):
    """Wraps the mock and counts every request that reaches the wire."""

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self.inner = inner
        self.requests: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(f"{request.method} {request.url.path}")
        return await self.inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self.inner.aclose()


def _client() -> tuple[httpx.AsyncClient, _CountingTransport]:
    app = make_app("openai_clean")
    counting = _CountingTransport(httpx.ASGITransport(app=app))
    return (
        httpx.AsyncClient(transport=counting, base_url="https://mock.test"),
        counting,
    )


async def test_evaluate_sends_nothing_before_the_estimate(tmp_path: Path) -> None:
    seen: list[Estimate] = []
    client, counting = _client()

    def refuse(estimate: Estimate) -> bool:
        # Anything already sent at this point was spent without consent.
        assert counting.requests == [], counting.requests
        seen.append(estimate)
        return False

    try:
        result = await aevaluate_target(
            "https://mock.test/v1/chat/completions",
            key="test-key-abcdefgh", client=client, root=str(tmp_path), runs=2,
            authorized=True, authorization_prompt=False, confirm=refuse,
        )
    finally:
        await client.aclose()

    assert seen, "the estimate was never shown"
    assert result.declined
    assert counting.requests == [], counting.requests
    assert not (tmp_path / "runs").exists()


async def test_the_evaluate_estimate_covers_discovery_too(tmp_path: Path) -> None:
    """Discovery and capability detection spend first; gating after them is
    gating after the money is gone."""
    captured: list[Estimate] = []
    client, _ = _client()

    def refuse(estimate: Estimate) -> bool:
        captured.append(estimate)
        return False

    try:
        await aevaluate_target(
            "https://mock.test/v1/chat/completions",
            key="test-key-abcdefgh", client=client, root=str(tmp_path), runs=2,
            authorized=True, authorization_prompt=False, confirm=refuse,
        )
    finally:
        await client.aclose()

    phases = {p.phase for p in captured[0].phases}
    assert "discovery" in phases
    assert "capabilities" in phases


async def test_a_cap_below_discovery_stops_evaluate_before_it_starts(
    tmp_path: Path,
) -> None:
    client, counting = _client()
    try:
        result = await aevaluate_target(
            "https://mock.test/v1/chat/completions",
            key="test-key-abcdefgh", client=client, root=str(tmp_path), runs=2,
            authorized=True, authorization_prompt=False,
            cap=BudgetCap(value=5, unit="requests"),
        )
    finally:
        await client.aclose()

    assert result.declined
    assert "nothing was sent" in result.declined
    assert counting.requests == [], counting.requests


async def test_the_sweep_path_still_sends_nothing_before_consent(
    tmp_path: Path,
) -> None:
    client, counting = _client()

    def refuse(estimate: Estimate) -> bool:
        assert counting.requests == [], counting.requests
        return False

    try:
        result = await asweep_target(
            "https://mock.test/v1/chat/completions",
            key="test-key-abcdefgh", client=client, root=str(tmp_path), runs=2,
            authorized=True, authorization_prompt=False, confirm=refuse,
        )
    finally:
        await client.aclose()

    assert result.status.value == "DECLINED"
    assert counting.requests == [], counting.requests


@pytest.mark.parametrize("verb", ["evaluate", "baseline", "gate"])
def test_every_spending_verb_offers_yes(verb: str) -> None:
    """Without a TTY and without --yes the run declines, so each verb that
    spends has to expose the flag or be unusable in CI."""
    import inspect

    from sweepeval.cli import evaluate as cli

    command = getattr(cli, f"{verb}_command")
    assert "yes" in inspect.signature(command).parameters, verb
