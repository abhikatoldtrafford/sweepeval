"""What each capability detector can actually distinguish (spec §9).

Two of the eight detectors turned out to assert a property of the target from
a probe that could not elicit it: the tool probe never offered a tool, and the
citation probe asked for sources with no answer to source and looked for keys
a real retrieval endpoint does not use. Both reported UNSUPPORTED against
endpoints that supported the capability, and both deferred a scorer family on
that reading.

This file audits the three that were left, with one question each: against a
target that genuinely has the capability, and one that genuinely lacks it,
does the probe return different answers -- and for the right reason?

It found two more defects and a third that made one of them invisible.
"""

from __future__ import annotations

import httpx
import pytest

from sweepeval.capabilities.detect import (
    CapabilityBudget,
    Support,
    detect_multi_turn,
    detect_refusal_baseline,
    detect_system_prompt,
)
from sweepeval.discovery.ladder import climb
from sweepeval.mock.app import MockApp
from sweepeval.mock.scenario import Scenario

URL = "https://mock.test/v1/chat/completions"
TEXT_PATH = "$.choices[0].message.content"


def scenario(**kw) -> Scenario:
    base = dict(
        name="probe", description="audit fixture", shape="openai",
        paths=("/v1/chat/completions",), auth="none",
    )
    base.update(kw)
    return Scenario.model_validate(base)


async def _probe(spec: Scenario, detector):
    app = MockApp(spec)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://mock.test"
    )
    try:
        ladder = await climb(client, URL, None)
        return await detector(client, ladder, None, CapabilityBudget(), TEXT_PATH)
    finally:
        await client.aclose()


# --- system prompt: "it changed" is not "the system role did it" -------------


async def test_a_target_that_honours_the_role_is_supported() -> None:
    result = await _probe(
        scenario(supports_system_prompt=True), detect_system_prompt
    )
    assert result.support is Support.SUPPORTED
    assert result.evidence["marker_honoured"] is True


async def test_a_target_that_drops_the_role_is_unsupported() -> None:
    """The case the detector exists for: otherwise the sweep gets a
    system-prompt axis whose four variants are four identical configs."""
    result = await _probe(
        scenario(supports_system_prompt=False), detect_system_prompt
    )
    assert result.support is Support.UNSUPPORTED
    assert "dropped" in result.evidence["reason"]


async def test_a_noisy_target_that_drops_the_role_is_not_called_supported() -> None:
    """The defect. `output_changed` conflated "the system prompt had an
    effect" with "this target never says the same thing twice", so a target
    that silently dropped the role reported SUPPORTED on nothing but its own
    noise -- and nondeterminism is not rare: the published scorecard found no
    model in fourteen reproducible at temperature 0.

    Fixed with a control, the same shape of fix the degradation family's load
    metric needed: measure the target's own variability, and refuse to
    attribute a difference to the condition when two identical requests differ
    too.
    """
    result = await _probe(
        scenario(supports_system_prompt=False, nondeterministic_at_temp0=True),
        detect_system_prompt,
    )
    assert result.support is not Support.SUPPORTED
    assert result.support is Support.INCONCLUSIVE
    assert result.evidence["marker_honoured"] is False
    assert result.evidence["target_is_steady"] is False
    assert "cannot be attributed" in result.evidence["reason"]


async def test_a_noisy_target_that_honours_the_role_is_still_supported() -> None:
    """The control must not cost the detector its true positives."""
    result = await _probe(
        scenario(supports_system_prompt=True, nondeterministic_at_temp0=True),
        detect_system_prompt,
    )
    assert result.support is Support.SUPPORTED
    assert result.evidence["marker_honoured"] is True


# --- refusal baseline: the most structured way of declining ------------------


async def test_a_refusal_in_the_content_field_is_recognised() -> None:
    result = await _probe(
        scenario(refuses_everything=True), detect_refusal_baseline
    )
    assert result.support is Support.SUPPORTED
    assert result.evidence["markers"]


async def test_a_structured_refusal_is_recognised() -> None:
    """OpenAI returns `content: null` with `refusal` populated. The detector
    read only the content path, so the probe came back empty and the detector
    whose entire job is recognising how a target declines could not see the
    most structured way of declining there is.

    The runner had learned this after a live run lost five of gpt-5.1's 72
    security trials to it. The fix had landed at one call site of two.
    """
    result = await _probe(
        scenario(refuses_everything=True, structured_refusal=True),
        detect_refusal_baseline,
    )
    assert result.support is Support.SUPPORTED
    assert result.evidence["markers"]
    assert result.evidence["sample_length"] > 0


async def test_a_target_that_complies_shows_no_refusal_language() -> None:
    """INCONCLUSIVE has to stay reachable, or the fix has only moved the bias.

    A default scenario is not the fixture for this: the mock declines the
    disallowed probe the way any well-behaved target would, so it is a
    SUPPORTED case. The target that teaches the detector nothing is one that
    complies with the probe.
    """
    result = await _probe(scenario(leaks_guardrails=True), detect_refusal_baseline)
    assert result.support is Support.INCONCLUSIVE
    assert result.evidence["markers"] == []
    assert "may be indistinguishable" in result.evidence["reason"]


# --- multi-turn: the fixture that never existed ------------------------------


async def test_a_target_that_carries_history_is_supported() -> None:
    result = await _probe(scenario(), detect_multi_turn)
    assert result.support is Support.SUPPORTED
    assert result.evidence["fact_recalled"] is True


async def test_a_target_that_rejects_history_is_unsupported() -> None:
    """`supports_multi_turn` was declared on the Scenario and read nowhere, so
    there was no fixture for a target without conversation support and this
    detector had never been run against one. With the knob dead it reported
    SUPPORTED -- the mock answered normally, because nothing was disabled."""
    result = await _probe(scenario(supports_multi_turn=False), detect_multi_turn)
    assert result.support is Support.UNSUPPORTED
    assert "rejected" in result.evidence["reason"]


async def test_a_target_that_forgets_is_not_claimed_as_supported() -> None:
    result = await _probe(scenario(context_drop_depth=1), detect_multi_turn)
    assert result.support is not Support.SUPPORTED
    assert result.evidence["fact_recalled"] is False


# --- the class of defect, not the instances ----------------------------------


def test_every_scenario_knob_changes_what_the_mock_does() -> None:
    """A test double with a knob that does nothing is a fixture that lies.

    Six of forty fields were read nowhere: `supports_streaming`,
    `emit_usage_when_streaming`, `supports_multi_turn`, `fail_paths`,
    `error_rate` -- and the scenario named `anthropic_streaming` set one of
    them while the mock had no streaming code at all.

    The cost is not tidiness. `supports_multi_turn` being dead meant the
    multi-turn detector had no negative fixture, so the one test that would
    have caught it could never have been written.
    """
    import pathlib

    app = pathlib.Path(
        MockApp.__module__.replace(".", "/")
    )
    source = (pathlib.Path(__file__).parents[2] / "src" / app).with_suffix(".py")
    text = source.read_text(encoding="utf-8")

    metadata = {"name", "description", "extra"}
    dead = [
        field
        for field in Scenario.model_fields
        if field not in metadata and f"scenario.{field}" not in text
    ]
    assert not dead, f"declared and read nowhere: {dead}"


@pytest.mark.parametrize(
    ("knob", "request_body", "expect_status"),
    [
        (
            {"fail_paths": ("/v1/chat/completions",)},
            {"messages": [{"role": "user", "content": "hi"}]},
            500,
        ),
        (
            {"supports_streaming": False},
            {"messages": [{"role": "user", "content": "hi"}], "stream": True},
            400,
        ),
    ],
)
async def test_a_revived_knob_actually_bites(knob, request_body, expect_status) -> None:
    app = MockApp(scenario(**knob))
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://mock.test"
    )
    try:
        response = await client.post("/v1/chat/completions", json=request_body)
    finally:
        await client.aclose()
    assert response.status_code == expect_status


async def test_the_mock_can_stream() -> None:
    """It could not, at all, while shipping a scenario called
    `anthropic_streaming` -- so the transport's SSE decoding had no fixture
    behind it."""
    from sweepeval.http.streaming import decode_sse, reassemble

    app = MockApp(scenario(supports_streaming=True))
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://mock.test"
    )
    try:
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
    finally:
        await client.aclose()

    assert response.status_code == 200
    assert "event-stream" in response.headers["content-type"]
    events = list(decode_sse([response.content]))
    assert events
    assert reassemble(events).strip()


async def test_the_system_prompt_probe_sends_its_control() -> None:
    """Asserted on the requests, not the verdict.

    Deleting the control call changes no verdict on a noisy target: "no
    control" and "a control that disagreed" both mean "not steady", so both
    land on INCONCLUSIVE. Only counting what went out distinguishes a detector
    that measured the target's variability from one that merely failed to.

    Three POSTs: with the system role, without it, and without it again. The
    second and third must be byte-identical, or the control is measuring
    something other than the target's own noise.
    """
    import json

    bodies: list[bytes] = []

    class _Recording(httpx.ASGITransport):
        async def handle_async_request(self, request: httpx.Request):
            if request.method == "POST":
                bodies.append(request.content)
            return await super().handle_async_request(request)

    app = MockApp(scenario(supports_system_prompt=False))
    client = httpx.AsyncClient(
        transport=_Recording(app=app), base_url="https://mock.test"
    )
    try:
        ladder = await climb(client, URL, None)
        bodies.clear()  # discovery's own probes are not the subject
        await detect_system_prompt(
            client, ladder, None, CapabilityBudget(), TEXT_PATH
        )
    finally:
        await client.aclose()

    assert len(bodies) == 3, [json.loads(b) for b in bodies]
    assert bodies[1] == bodies[2], "the control is not the same request"
    assert bodies[0] != bodies[1], "the system role never varied"
