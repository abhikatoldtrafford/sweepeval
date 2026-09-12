"""Capability detection against the mock (spec §9). I5 and I8 load-bearing."""

from __future__ import annotations

from tests.conftest import make_app, make_client

from sweepeval.capabilities.detect import (
    DEFAULT_CAPABILITY_REQUESTS,
    Capability,
    CapabilityBudget,
    Support,
    detect_all,
)
from sweepeval.discovery.runner import discover_target


async def _capabilities(name: str, key: str | None = None, profile: str = "standard"):
    app = make_app(name)
    client = make_client(app)
    try:
        url = "https://mock.test" + app.scenario.paths[0]
        outcome = await discover_target(client, url, key, seed=7)
        return await detect_all(
            client, outcome.ladder, key, outcome.extraction.path, profile=profile
        )
    finally:
        await client.aclose()


# --- every result feeds a SKIPPED reason (I5) ------------------------------


async def test_every_capability_records_verdict_method_and_evidence() -> None:
    report = await _capabilities("openai_clean", key="test-key-abcdefgh")
    manifest = report.to_manifest()
    for name, entry in manifest.items():
        assert entry["verdict"], name
        assert entry["method"], name
        assert entry["confidence"], name


async def test_a_capability_verdict_supplies_a_scorer_skip_reason() -> None:
    """I5: a scorer never silently vanishes; it names the detector."""
    report = await _capabilities("openai_clean", key="test-key-abcdefgh")
    reason = report.skip_reason(Capability.TOOL_CALLING)
    assert "tool_calling=" in reason
    assert "probe" in reason


async def test_an_unprobed_capability_still_gives_a_reason() -> None:
    report = await _capabilities("openai_clean", key="test-key-abcdefgh")
    assert "NOT_PROBED" in report.skip_reason(Capability.CONTEXT_CEILING)


# --- system prompt ---------------------------------------------------------


async def test_a_target_that_ignores_the_system_role_is_unsupported() -> None:
    """Accepting a system role is not honouring one.

    The mock accepts the messages array but its reply does not change, so the
    detector must not hand the sweep a four-variant axis that produces four
    identical configs.
    """
    report = await _capabilities("openai_clean", key="test-key-abcdefgh")
    result = report[Capability.SYSTEM_PROMPT]
    assert result.support is Support.UNSUPPORTED
    assert "dropped" in result.evidence["reason"]


# --- multi-turn ------------------------------------------------------------


async def test_multi_turn_reports_the_history_was_not_used() -> None:
    report = await _capabilities("openai_clean", key="test-key-abcdefgh")
    result = report[Capability.MULTI_TURN]
    assert result.evidence["history_accepted"] is True
    assert result.evidence["transport"] == "stateless_replay"


# --- refusal baseline ------------------------------------------------------


async def test_a_refusing_target_yields_a_refusal_fingerprint() -> None:
    """§11.8 needs this: without it a refusal is indistinguishable from an error."""
    report = await _capabilities("refuses_everything")
    result = report[Capability.REFUSAL_BASELINE]
    assert result.support is Support.SUPPORTED
    assert result.evidence["markers"]
    assert result.evidence["fingerprint"]


async def test_a_compliant_target_has_no_refusal_language() -> None:
    report = await _capabilities("openai_clean", key="test-key-abcdefgh")
    result = report[Capability.REFUSAL_BASELINE]
    assert result.support is Support.INCONCLUSIVE
    assert "indistinguishable" in result.evidence["reason"]


# --- tool calling and retrieval --------------------------------------------


async def test_a_plain_endpoint_exposes_no_tool_structure() -> None:
    report = await _capabilities("openai_clean", key="test-key-abcdefgh")
    assert report[Capability.TOOL_CALLING].support is Support.UNSUPPORTED
    assert report[Capability.RETRIEVAL].support is Support.UNSUPPORTED


# --- I8: the capability budget ---------------------------------------------


async def test_the_capability_phase_has_its_own_cap() -> None:
    assert DEFAULT_CAPABILITY_REQUESTS == 60


async def test_detection_stays_inside_its_budget() -> None:
    app = make_app("openai_clean")
    client = make_client(app)
    budget = CapabilityBudget(max_posts=3)
    try:
        outcome = await discover_target(
            client, "https://mock.test/v1/chat/completions", "test-key-abcdefgh", seed=7
        )
        report = await detect_all(
            client, outcome.ladder, "test-key-abcdefgh", outcome.extraction.path,
            budget=budget,
        )
    finally:
        await client.aclose()
    assert budget.posts <= 3
    # Detectors that could not run say so rather than claiming a verdict.
    inconclusive = [
        r for r in report.results.values() if r.support is Support.INCONCLUSIVE
    ]
    assert inconclusive


async def test_the_context_ceiling_search_never_runs_outside_deep() -> None:
    """§9: the largest unbudgeted spend in the tool."""
    for profile in ("quick", "standard"):
        report = await _capabilities("openai_clean", key="test-key-abcdefgh",
                                     profile=profile)
        assert report[Capability.CONTEXT_CEILING].support is Support.NOT_PROBED


# --- the detectors are cheap ----------------------------------------------


async def test_detection_costs_a_bounded_number_of_requests() -> None:
    app = make_app("openai_clean")
    client = make_client(app)
    budget = CapabilityBudget()
    try:
        outcome = await discover_target(
            client, "https://mock.test/v1/chat/completions", "test-key-abcdefgh", seed=7
        )
        await detect_all(
            client, outcome.ladder, "test-key-abcdefgh", outcome.extraction.path,
            budget=budget,
        )
    finally:
        await client.aclose()
    assert budget.posts <= 10, [a.render() for a in budget.attempts]
