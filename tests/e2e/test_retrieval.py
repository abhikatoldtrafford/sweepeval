"""Retrieval, end to end (spec §11, family 6).

The last deferred family, and the second whose deferral rested on a detector
that could not return the answer it was looking for. The citation probe asked
"What sources support your answer? Cite them." -- with no prior answer to
support -- and then searched the response for
`documents`/`sources`/`citations`/`retrieved`/`chunks`.

Measured against `gpt-5-search-api`, a live search-backed model that returns
real citations: not one of those keys appears anywhere in its response. Its
sources arrive as `message.annotations[].url_citation`, so the detector
reported `retrieval=UNSUPPORTED` for a working retrieval endpoint,
indistinguishable from a plain model.

The family it unblocks is shaped by one constraint the others do not have:
**sweepeval cannot supply the corpus.** precision@k, recall@k, MRR and nDCG all
need relevance labels over the target's own documents, so they are reported
SKIPPED with that reason rather than approximated. That is not a gap waiting
on effort -- it is outside what a black-box measurement can say, and the tests
below pin that it keeps saying so.
"""

from __future__ import annotations

import tempfile

import pytest
from tests.conftest import make_app, make_client

from sweepeval.execute.sweep import asweep_target
from sweepeval.retrieval import extract_citations, validate_citation
from sweepeval.schema.observation import Verdict
from sweepeval.scorers.retrieval import (
    INTEGRITY_METRIC,
    STABILITY_METRIC,
    UNMEASURABLE,
)


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
        if o.family == "retrieval" and o.metric == metric
    ]


# --- the detector that deferred the family -----------------------------------


async def test_the_probe_asks_something_that_needs_retrieving() -> None:
    """The first half of the defect. "What sources support your answer?" has
    no answer in front of it, so nothing about it requires going and looking.
    """
    from sweepeval.capabilities.detect import _RETRIEVAL_PROBE

    assert "support your answer" not in _RETRIEVAL_PROBE.lower()
    # It has to name something a model cannot know from weights alone.
    assert any(
        cue in _RETRIEVAL_PROBE.lower()
        for cue in ("most recent", "current", "latest")
    ), _RETRIEVAL_PROBE


@pytest.mark.parametrize(
    "scenario",
    ["cites_via_annotations", "cites_via_anthropic", "cites_via_documents"],
)
async def test_a_structured_channel_is_detected(scenario: str) -> None:
    from sweepeval.capabilities.detect import Capability, Support

    result = await _sweep(scenario, profile="quick")
    verdict = result.capabilities.results[Capability.RETRIEVAL]
    assert verdict.support is Support.SUPPORTED, verdict.evidence


async def test_the_annotations_channel_is_the_one_that_was_missed() -> None:
    """The regression in one assertion. This is the shape a live
    search-backed model returns, and the old key set contained none of it."""
    from sweepeval.capabilities.detect import Capability, Support

    result = await _sweep("cites_via_annotations", profile="quick")
    verdict = result.capabilities.results[Capability.RETRIEVAL]
    assert verdict.support is Support.SUPPORTED
    assert verdict.evidence["channels"] == ["openai.annotations"]


async def test_a_plain_model_is_still_unsupported() -> None:
    """UNSUPPORTED has to stay reachable, or the fix has only moved the bias."""
    from sweepeval.capabilities.detect import Capability, Support

    result = await _sweep("cites_nothing", profile="quick")
    verdict = result.capabilities.results[Capability.RETRIEVAL]
    assert verdict.support is Support.UNSUPPORTED
    assert verdict.evidence["citations"] == 0


async def test_links_in_prose_are_not_retrieval() -> None:
    """Measured live: gpt-4.1-nano answers the probe with a URL written into
    its text and no retrieval behind it. Crediting that would report a
    capability it does not have."""
    from sweepeval.capabilities.detect import Capability, Support

    result = await _sweep("cites_via_inline", profile="quick")
    verdict = result.capabilities.results[Capability.RETRIEVAL]
    assert verdict.support is Support.UNSUPPORTED
    assert verdict.evidence["cited_in_text_only"] is True


# --- five channels, one normalised citation ----------------------------------


@pytest.mark.parametrize(
    ("scenario", "channel"),
    [
        ("cites_via_annotations", "openai.annotations"),
        ("cites_via_anthropic", "anthropic.citations"),
        ("cites_via_gemini", "gemini.grounding"),
        ("cites_via_documents", "generic.documents"),
        ("cites_via_inline", "text.inline"),
    ],
)
async def test_each_channel_is_read(scenario: str, channel: str) -> None:
    import httpx

    app = make_app(scenario)
    prompt = "What is the current Bank of England base rate? Cite the source."
    if app.scenario.shape == "gemini":
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    else:
        body = {"messages": [{"role": "user", "content": prompt}]}
        if app.scenario.shape == "anthropic":
            body["max_tokens"] = 256

    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://mock.test"
    )
    try:
        response = await client.post(app.scenario.paths[0], json=body)
    finally:
        await client.aclose()

    found = extract_citations(response.json())
    assert found, f"{scenario}: nothing decoded"
    assert found[0].channel == channel
    assert found[0].source


def test_a_structured_citation_beats_a_link_in_the_prose() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": "See https://example.test/prose",
                    "annotations": [
                        {"type": "url_citation",
                         "url_citation": {"url": "https://example.test/real"}}
                    ],
                }
            }
        ]
    }
    found = extract_citations(payload, "See https://example.test/prose")
    assert [c.channel for c in found] == ["openai.annotations"]
    assert found[0].source == "https://example.test/real"


# --- what can be checked without leaving the target --------------------------


def test_a_span_past_the_end_of_the_answer_is_malformed() -> None:
    from sweepeval.retrieval import Citation

    answer = "a short answer"
    bad = validate_citation(
        Citation(source="https://x.test", channel="openai.annotations",
                 start=0, end=len(answer) + 50),
        answer,
    )
    assert not bad.ok
    assert "past the" in bad.problems[0]

    good = validate_citation(
        Citation(source="https://x.test", channel="openai.annotations",
                 start=0, end=len(answer)),
        answer,
    )
    assert good.ok


def test_a_citation_that_identifies_nothing_is_malformed() -> None:
    from sweepeval.retrieval import Citation

    empty = validate_citation(Citation(source="", channel="generic.documents"), "x")
    assert not empty.ok
    assert "identifies no source" in empty.problems[0]


def test_nothing_in_the_module_fetches_a_citation() -> None:
    """sweepeval talks to the endpoint you named and to nothing else.

    Resolving a citation would mean issuing requests to third parties on a
    user's behalf, from a tool whose README promises it makes no other network
    calls -- so "this URL exists" and "this page supports the claim" are out of
    scope, and the report must not imply the citation was verified.

    Checked against the module's imports rather than its text: the first
    version of this grepped the source and tripped over the word "requests" in
    the docstring explaining why there are none.
    """
    import ast
    import inspect

    import sweepeval.retrieval as module

    tree = ast.parse(inspect.getsource(module))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    networking = {"httpx", "requests", "aiohttp", "socket", "urllib", "http"}
    assert not imported & networking, imported & networking


# --- the metrics that cannot exist -------------------------------------------


async def test_the_ir_metrics_are_reported_skipped_with_the_reason() -> None:
    """A reader of an "evaluation tool" looks for nDCG first. Omitting it is
    indistinguishable from a tool that computed it and found nothing."""
    result = await _sweep("cites_via_annotations")
    rows = {
        o.metric: o for o in result.store.observations.read()
        if o.family == "retrieval" and o.metric in UNMEASURABLE
    }
    assert set(rows) == set(UNMEASURABLE), sorted(rows)
    for metric, row in rows.items():
        assert row.verdict is Verdict.SKIPPED, metric
        assert "relevance labels" in (row.reason or ""), metric
        assert "cannot supply the corpus" in (row.reason or ""), metric


def test_the_ir_metrics_are_not_secretly_aggregated() -> None:
    """SKIPPED with a reason, not a number computed from an invented corpus."""
    from sweepeval.execute.aggregation import RATE_METRICS

    assert not set(UNMEASURABLE) & set(RATE_METRICS)


# --- integrity ---------------------------------------------------------------


async def test_a_grounded_answer_with_sound_citations_passes() -> None:
    result = await _sweep("cites_via_annotations")
    grounded = [
        o for o in _rows(result, INTEGRITY_METRIC)
        if o.unit_id.startswith("rt.grounded")
    ]
    assert grounded
    assert all(o.verdict is Verdict.PASS for o in grounded), [
        (o.unit_id, o.verdict.value, o.reason) for o in grounded
    ]


async def test_a_question_that_needed_a_source_and_got_none_fails() -> None:
    result = await _sweep("cites_nothing")
    if not _rows(result, INTEGRITY_METRIC):
        pytest.skip("the family is capability-gated and this target has none")
    grounded = [
        o for o in _rows(result, INTEGRITY_METRIC)
        if o.unit_id.startswith("rt.grounded")
    ]
    assert all(o.verdict is Verdict.FAIL for o in grounded)


async def test_fabricating_a_citation_is_the_failure_that_matters() -> None:
    """Four probes ask about a company that does not exist, a standard never
    published, an event that has not happened and a fact nobody could know."""
    result = await _sweep("fabricates_citations")
    invented = [
        o for o in _rows(result, INTEGRITY_METRIC)
        if o.unit_id.startswith("rt.unsourceable")
    ]
    assert invented
    assert all(o.verdict is Verdict.FAIL for o in invented), [
        (o.unit_id, o.verdict.value, o.reason) for o in invented
    ]
    assert any("nothing could source" in (o.reason or "") for o in invented)


async def test_declining_to_source_the_unsourceable_is_credited() -> None:
    """The same probes against a target that behaves. Without this the
    fabrication test passes on a scorer that fails those probes always."""
    result = await _sweep("cites_via_annotations")
    invented = [
        o for o in _rows(result, INTEGRITY_METRIC)
        if o.unit_id.startswith("rt.unsourceable")
    ]
    assert invented
    assert all(o.verdict is Verdict.PASS for o in invented), [
        (o.unit_id, o.verdict.value, o.reason) for o in invented
    ]


@pytest.mark.parametrize(
    ("scenario", "fragment"),
    [
        ("citations_without_a_source", "identifies no source"),
        ("citation_span_out_of_bounds", "past the"),
    ],
)
async def test_a_malformed_citation_fails_and_says_how(
    scenario: str, fragment: str
) -> None:
    result = await _sweep(scenario)
    failed = [o for o in _rows(result, INTEGRITY_METRIC) if o.verdict is Verdict.FAIL]
    assert failed, [
        (o.unit_id, o.verdict.value, o.reason) for o in _rows(result, INTEGRITY_METRIC)
    ]
    assert any(fragment in (o.reason or "") for o in failed), [o.reason for o in failed]


# --- stability ---------------------------------------------------------------


async def test_the_same_question_surfacing_the_same_sources_is_stable() -> None:
    result = await _sweep("cites_via_annotations")
    rows = _rows(result, STABILITY_METRIC)
    assert rows, "no stability rows; the cross-run pass did not reach this family"
    grounded = [o for o in rows if o.unit_id.startswith("rt.grounded")]
    assert grounded
    assert all(o.verdict is Verdict.PASS for o in grounded), [
        (o.unit_id, o.verdict.value, o.reason) for o in grounded
    ]


def test_sources_that_change_between_runs_are_unstable() -> None:
    from sweepeval.corpus.loader import load_corpus
    from sweepeval.scorers.base import RunEvidence, ScoreContext
    from sweepeval.scorers.retrieval import RetrievalScorer

    unit = load_corpus("standard").by_family("retrieval")[0].to_unit()

    def payload(url: str):
        return {
            "choices": [
                {"message": {"content": "answer", "annotations": [
                    {"type": "url_citation", "url_citation": {"url": url}}
                ]}}
            ]
        }

    context = ScoreContext(run_id="r", config_id="cfg-00", run_idx=0, text="")
    same = RetrievalScorer().finalize(
        [RunEvidence(unit=unit, texts=("a", "a"),
                     payloads=(payload("https://x.test/1"), payload("https://x.test/1")))],
        context,
    )
    assert same[0].verdict is Verdict.PASS

    differ = RetrievalScorer().finalize(
        [RunEvidence(unit=unit, texts=("a", "a"),
                     payloads=(payload("https://x.test/1"), payload("https://x.test/2")))],
        context,
    )
    assert differ[0].verdict is Verdict.FAIL
    assert "differ between runs" in (differ[0].reason or "")


# --- and it is no longer deferred --------------------------------------------


def test_no_family_is_deferred_any_more() -> None:
    from sweepeval.scorers import registry
    from sweepeval.scorers.deferred import DeferredScorer

    deferred = [s.family for s in registry().all() if isinstance(s, DeferredScorer)]
    assert not deferred, deferred


def test_the_deferral_mechanism_is_kept() -> None:
    """Two of the three families were deferred on capability readings that
    turned out to be wrong. Being loudly absent is what made them checkable,
    so the mechanism stays even with nothing using it."""
    from sweepeval.scorers.deferred import DEFERRED_REASON, DeferredScorer

    scorer = DeferredScorer(
        family="example", metric_name="example_metric", brief="not built"
    )
    rows = scorer.score(
        _any_unit(), [],
        __import__("sweepeval.scorers.base", fromlist=["ScoreContext"]).ScoreContext(
            run_id="r", config_id="c", run_idx=0, text=""
        ),
    )
    assert rows[0].verdict is Verdict.SKIPPED
    assert DEFERRED_REASON in (rows[0].reason or "")


def _any_unit():
    from sweepeval.corpus.loader import load_corpus

    return load_corpus("standard").probes[0].to_unit()
