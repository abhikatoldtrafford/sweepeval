"""Tool integrity, end to end (spec §11, family 4).

The family was deferred on a reading that turned out to be false. Its detector
sent a bare prompt -- "if you have a tool available, call it" -- and looked for
`tool_calls` in the reply, without ever offering a tool. A chat API only emits
that field when the request carries a `tools` array, so `UNSUPPORTED` was the
only answer it could ever give, and every run this tool has done reported it
against endpoints that support tool calling perfectly well.

So the first thing here is the detector, and the rest is the family it unblocks.

"Supports tool calling" is not one wire format, which is why four are
exercised. A scorer that read only `tool_calls` would score a target emitting
the deprecated `function_call` as having no tools at all; one that read text
would credit a model that merely *describes* a call. Both are wrong in
opposite directions, and only the four-way fixture catches both.
"""

from __future__ import annotations

import tempfile

import pytest
from tests.conftest import make_app, make_client

from sweepeval.execute.sweep import asweep_target
from sweepeval.schema.observation import Verdict
from sweepeval.scorers.tool_integrity import SELECTION_METRIC, VALIDITY_METRIC
from sweepeval.tools import TOOLKIT, extract_tool_calls, offer_for_shape, validate_call


async def _sweep(scenario: str, profile: str = "standard"):
    app = make_app(scenario)
    client = make_client(app)
    try:
        return await asweep_target(
            "https://mock.test" + app.scenario.paths[0],
            client=client, root=tempfile.mkdtemp(), runs=2, profile=profile,
            authorized=True, authorization_prompt=False, seed=7, config_cap=1,
        )
    finally:
        await client.aclose()


def _rows(result, metric: str):
    return [
        o for o in result.store.observations.read()
        if o.family == "tool_integrity" and o.metric == metric
    ]


# --- the detector, which is why the family was deferred ----------------------


async def test_a_tool_is_actually_offered() -> None:
    """The defect in one assertion: the probe must carry a tool declaration.

    Without this the detector asks a question whose answer is fixed, and every
    endpoint on earth is UNSUPPORTED.
    """
    import httpx

    from sweepeval.capabilities.detect import CapabilityBudget, detect_tool_calling
    from sweepeval.discovery.ladder import climb

    app = make_app("native_tool_calls")
    seen: list[dict] = []

    class _Recording(httpx.ASGITransport):
        async def handle_async_request(self, request):
            import contextlib
            import json as _json

            with contextlib.suppress(ValueError):
                seen.append(_json.loads(request.content or b"{}"))
            return await super().handle_async_request(request)

    client = httpx.AsyncClient(
        transport=_Recording(app=app), base_url="https://mock.test"
    )
    try:
        ladder = await climb(client, "https://mock.test/v1/chat/completions", None)
        await detect_tool_calling(client, ladder, None, CapabilityBudget())
    finally:
        await client.aclose()

    offered = [body for body in seen if body.get("tools")]
    assert offered, "the tool probe carried no tool declaration"
    assert {t["function"]["name"] for t in offered[-1]["tools"]} == {
        t.name for t in TOOLKIT
    }


@pytest.mark.parametrize(
    "scenario",
    ["native_tool_calls", "legacy_function_call", "anthropic_tools"],
)
async def test_a_structured_encoding_is_detected_as_supported(scenario: str) -> None:
    from sweepeval.capabilities.detect import Capability, Support

    result = await _sweep(scenario, profile="quick")
    verdict = result.capabilities.results[Capability.TOOL_CALLING]
    assert verdict.support is Support.SUPPORTED, verdict.evidence


async def test_a_target_that_ignores_the_offer_is_unsupported() -> None:
    """UNSUPPORTED is still reachable, and now it means something: the target
    was given a tool and did not use it."""
    from sweepeval.capabilities.detect import Capability, Support

    result = await _sweep("ignores_tool_offer", profile="quick")
    verdict = result.capabilities.results[Capability.TOOL_CALLING]
    assert verdict.support is Support.UNSUPPORTED
    assert verdict.evidence["tools_offered"], "unsupported without offering anything"


async def test_imitating_a_call_in_prose_is_not_support() -> None:
    """The distinction the old detector could not draw either way round. A
    model that writes `<tool_call>{...}` into its message has not done
    structured tool calling, and the report says so rather than crediting it."""
    from sweepeval.capabilities.detect import Capability, Support

    result = await _sweep("imitates_tools", profile="quick")
    verdict = result.capabilities.results[Capability.TOOL_CALLING]
    assert verdict.support is Support.UNSUPPORTED
    assert verdict.evidence["imitated_in_text"] is True


async def test_a_shape_that_cannot_carry_tools_is_not_probed() -> None:
    """NOT_PROBED, not UNSUPPORTED. A raw-text endpoint has not failed a tool
    probe; it cannot be given one, and saying otherwise would be a claim about
    the target rather than about us.

    Built directly rather than via a scenario: which shape a scenario resolves
    to is discovery's business, and a test that skips when it resolves
    differently is a test that can stop checking without anyone noticing.
    """

    from sweepeval.capabilities.detect import (
        Capability,
        CapabilityBudget,
        Support,
        detect_tool_calling,
    )
    from sweepeval.discovery.ladder import climb

    app = make_app("weird_shape")
    client = make_client(app)
    try:
        ladder = await climb(client, "https://mock.test" + app.scenario.paths[0], None)
        # Force the shape whose defining property is that it cannot carry a
        # tool declaration.
        from sweepeval.discovery.shapes.base import registry as shape_registry

        ladder.shape = shape_registry()["raw.text"]
        assert not offer_for_shape(ladder.shape.name)
        verdict = await detect_tool_calling(client, ladder, None, CapabilityBudget())
    finally:
        await client.aclose()

    assert verdict.capability is Capability.TOOL_CALLING
    assert verdict.support is Support.NOT_PROBED
    assert "nothing was offered" in verdict.evidence["reason"]


# --- the four encodings decode to the same thing -----------------------------


@pytest.mark.parametrize(
    ("scenario", "encoding"),
    [
        ("native_tool_calls", "openai.tool_calls"),
        ("legacy_function_call", "openai.function_call"),
        ("anthropic_tools", "anthropic.tool_use"),
        ("imitates_tools", "text.embedded"),
    ],
)
async def test_each_encoding_is_read(scenario: str, encoding: str) -> None:
    import httpx

    app = make_app(scenario)
    shape = (
        "anthropic.messages"
        if app.scenario.shape == "anthropic"
        else "openai.chat_completions"
    )
    body = {
        "messages": [{"role": "user", "content": "What is the status of order 48812?"}],
        **offer_for_shape(shape),
    }
    if shape == "anthropic.messages":
        body["max_tokens"] = 256

    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://mock.test"
    )
    try:
        response = await client.post(app.scenario.paths[0], json=body)
    finally:
        await client.aclose()

    calls = [validate_call(c) for c in extract_tool_calls(response.json())]
    assert calls, f"{scenario}: nothing decoded"
    assert calls[0].encoding == encoding
    assert calls[0].name == "lookup_order"
    assert calls[0].ok, calls[0].problems


def test_a_structured_call_beats_one_written_in_prose() -> None:
    """A target doing both should be credited with the real one."""
    payload = {
        "choices": [
            {
                "message": {
                    "content": '<tool_call>{"name": "get_utc_time"}</tool_call>',
                    "tool_calls": [
                        {"function": {"name": "lookup_order",
                                      "arguments": '{"order_id": "1"}'}}
                    ],
                }
            }
        ]
    }
    calls = extract_tool_calls(payload, payload["choices"][0]["message"]["content"])
    assert [c.encoding for c in calls] == ["openai.tool_calls"]


# --- validation against the schema we sent -----------------------------------


@pytest.mark.parametrize(
    ("scenario", "fragment"),
    [
        ("malformed_tool_json", "not valid JSON"),
        ("missing_required_argument", "missing required argument"),
        ("wrong_argument_type", "schema says"),
        ("hallucinates_a_tool", "no tool named"),
    ],
)
async def test_a_broken_call_fails_and_says_how(scenario: str, fragment: str) -> None:
    """Every one of these is decidable only because sweepeval sent the schema.
    None of it depends on knowing anything about the target."""
    result = await _sweep(scenario)
    failed = [o for o in _rows(result, VALIDITY_METRIC) if o.verdict is Verdict.FAIL]
    assert failed, [
        (o.unit_id, o.verdict.value, o.reason) for o in _rows(result, VALIDITY_METRIC)
    ]
    assert any(fragment in (o.reason or "") for o in failed), [
        o.reason for o in failed
    ]


async def test_a_well_behaved_target_passes() -> None:
    """Without this the file passes on a scorer that only ever fails.

    Asserted on a probe that *expects* a call, not on any pass at all: the
    restraint probes pass by the target doing nothing, so "some row passed"
    holds even when no tool was ever offered and nothing could be called.
    """
    result = await _sweep("native_tool_calls")
    rows = _rows(result, VALIDITY_METRIC)
    assert rows
    selecting = [o for o in rows if o.unit_id.startswith("ti.select")]
    assert selecting, "no probe expecting a tool ran"
    assert any(o.verdict is Verdict.PASS for o in selecting), [
        (o.unit_id, o.verdict.value, o.reason) for o in selecting
    ]
    assert any("with arguments the schema accepts" in (o.reason or "")
               for o in selecting)


async def test_calling_a_tool_nothing_asked_for_is_a_failure() -> None:
    """Over-calling is a real failure mode and it bills for the privilege. It
    is FAIL rather than UNSCORABLE: the target did something wrong, and that is
    not an inability to measure."""
    result = await _sweep("over_calls_tools")
    restraint = [
        o for o in _rows(result, VALIDITY_METRIC)
        if o.unit_id.startswith("ti.restraint")
    ]
    assert restraint
    assert any(
        o.verdict is Verdict.FAIL and "no tool was needed" in (o.reason or "")
        for o in restraint
    ), [(o.unit_id, o.verdict.value, o.reason) for o in restraint]


async def test_restraint_is_credited_when_it_is_earned() -> None:
    result = await _sweep("native_tool_calls")
    restraint = [
        o for o in _rows(result, VALIDITY_METRIC)
        if o.unit_id.startswith("ti.restraint")
    ]
    assert restraint
    assert all(o.verdict is Verdict.PASS for o in restraint), [
        (o.unit_id, o.verdict.value, o.reason) for o in restraint
    ]


async def test_a_target_that_only_imitates_never_reaches_the_family() -> None:
    """Coherence between the two layers, which is the honest outcome.

    Imitation is UNSUPPORTED at the capability layer, so `tool_integrity` is
    SKIPPED with that reason rather than scored -- I5, and the same rule every
    capability-gated family follows. The finding is not lost: it is in the
    capability evidence, which is where a user can act on it.
    """
    result = await _sweep("imitates_tools")
    assert not _rows(result, VALIDITY_METRIC)

    reasons = {family: reason for family, reason in result.skipped}
    assert "tool_integrity" in reasons
    assert "tool_calling=UNSUPPORTED" in reasons["tool_integrity"]


def test_a_call_written_in_prose_is_not_credited_as_one() -> None:
    """The scorer's own branch, reachable when a target emits structurally
    some of the time and writes prose the rest -- which the capability probe,
    being a single request, cannot rule out.

    Crediting it would report a capability the target does not have; scoring
    it as silence would lose the most useful thing the run found.
    """
    from sweepeval.corpus.loader import load_corpus
    from sweepeval.scorers.base import ScoreContext
    from sweepeval.scorers.tool_integrity import ToolIntegrityScorer

    unit = next(
        t for t in load_corpus("standard").by_family("tool_integrity")
        if t.expects_tool == "lookup_order"
    ).to_unit()
    prose = '<tool_call>{"name": "lookup_order", "arguments": {"order_id": "48812"}}</tool_call>'
    rows = ToolIntegrityScorer().score(
        unit, [],
        ScoreContext(
            run_id="r", config_id="cfg-00", run_idx=0, text=prose,
            payload={"choices": [{"message": {"content": prose}}]},
        ),
    )
    assert rows[0].verdict is Verdict.UNSCORABLE
    assert "imitation, not tool calling" in (rows[0].reason or "")
    assert rows[0].value is None


# --- selection stability is its own question ---------------------------------


async def test_selection_stability_is_measured_separately() -> None:
    result = await _sweep("native_tool_calls")
    rows = _rows(result, SELECTION_METRIC)
    assert rows, "no selection rows; the cross-run pass did not reach this family"
    assert all(o.verdict is not Verdict.SKIPPED for o in rows)


async def test_a_consistent_target_is_stable() -> None:
    """Asserted on probes that actually called something.

    The restraint probes are stable at "(none)", so "some row passed" holds
    even when every tool-calling run was excluded -- which is exactly what
    happened on this family's first live run: a reply carrying only tool calls
    has `content: null`, the cross-run pass required text, and
    `tool_selection_stability` came back "fewer than two scorable runs" on
    every probe that worked.
    """
    result = await _sweep("native_tool_calls")
    rows = _rows(result, SELECTION_METRIC)
    calling = [o for o in rows if o.unit_id.startswith("ti.select")]
    assert calling, "no tool-selecting probe produced a selection row"

    excluded = [o for o in calling if o.verdict is Verdict.UNSCORABLE]
    assert not excluded, [(o.unit_id, o.reason) for o in excluded]
    assert all(o.verdict is Verdict.PASS for o in calling), [
        (o.unit_id, o.verdict.value, o.reason) for o in calling
    ]
    # And the reason names the tool, not "(none)".
    assert any("openai.tool_calls:" in (o.reason or "") for o in calling), [
        o.reason for o in calling
    ]


def test_calling_nothing_twice_counts_as_stable() -> None:
    """Dropping the empty case would score an intermittent caller as perfectly
    consistent, which is the opposite of what the metric is for."""
    from sweepeval.corpus.loader import load_corpus
    from sweepeval.scorers.base import RunEvidence, ScoreContext
    from sweepeval.scorers.tool_integrity import ToolIntegrityScorer

    unit = next(
        t for t in load_corpus("standard").by_family("tool_integrity")
        if t.expects_tool == ""
    ).to_unit()
    silent = {"choices": [{"message": {"content": "no tool needed"}}]}
    once = {
        "choices": [
            {"message": {"tool_calls": [
                {"function": {"name": "get_utc_time", "arguments": "{}"}}
            ]}}
        ]
    }
    scorer = ToolIntegrityScorer()
    context = ScoreContext(run_id="r", config_id="cfg-00", run_idx=0, text="")

    stable = scorer.finalize(
        [RunEvidence(unit=unit, texts=("", ""), payloads=(silent, silent))], context
    )
    assert stable[0].verdict is Verdict.PASS

    flipped = scorer.finalize(
        [RunEvidence(unit=unit, texts=("", ""), payloads=(silent, once))], context
    )
    assert flipped[0].verdict is Verdict.FAIL
    assert "then" in (flipped[0].reason or "")


def test_selection_says_so_when_it_cannot_see_the_choice() -> None:
    """A resumed run carries no parsed responses. Reading that as "chose
    nothing on every run" would score it a perfect 1.0."""
    from sweepeval.corpus.loader import load_corpus
    from sweepeval.scorers.base import RunEvidence, ScoreContext
    from sweepeval.scorers.tool_integrity import ToolIntegrityScorer

    unit = load_corpus("standard").by_family("tool_integrity")[0].to_unit()
    rows = ToolIntegrityScorer().finalize(
        [RunEvidence(unit=unit, texts=("", ""), payloads=(None, None))],
        ScoreContext(run_id="r", config_id="cfg-00", run_idx=0, text=""),
    )
    assert rows[0].verdict is Verdict.UNSCORABLE
    assert "unknown" in (rows[0].reason or "")


# --- and it reaches the report -----------------------------------------------


async def test_the_metrics_aggregate_with_intervals() -> None:
    result = await _sweep("native_tool_calls")
    metrics = result.configs[0].metrics
    assert VALIDITY_METRIC in metrics
    value = metrics[VALIDITY_METRIC]
    assert value.point is not None
    assert (value.lo, value.hi) != (None, None), "a point with no interval"


def test_it_is_no_longer_a_deferred_family() -> None:
    from sweepeval.scorers import registry
    from sweepeval.scorers.deferred import DeferredScorer

    scorer = registry().get("tool_integrity")
    assert not isinstance(scorer, DeferredScorer)
    assert {m.metric for m in scorer.metrics()} == {VALIDITY_METRIC, SELECTION_METRIC}
