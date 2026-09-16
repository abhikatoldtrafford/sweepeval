"""The degradation family, end to end (spec §11, family 8).

Three dimensions were specified. Two are built and one says why it is not:

  long inputs   a planted fact recalled from under a growing haystack
  load          the same recall question asked under real contention
  tool failure  needs `tool_calling`, UNSUPPORTED on every endpoint measured

The assertions that matter are not "it produces numbers". They are that the
family can *fail*, that it can *pass*, and that it tells three things apart
which all look like a bad answer from the outside: the target forgot; the
target could not accept a body that size; the target throttled us.

And one property the rest of the tool depends on: the ramp is the only place
sweepeval contends with itself, so its calls must stay out of the latency
population. The operational scorer scores every unit's calls, including this
family's, so nothing else stops them.
"""

from __future__ import annotations

import tempfile

import pytest
from tests.conftest import make_app, make_client

from sweepeval.execute.sweep import asweep_target
from sweepeval.schema.observation import Verdict
from sweepeval.scorers.degradation import LOAD_METRIC, LONG_INPUT_METRIC


async def _sweep(scenario: str, profile: str = "standard"):
    app = make_app(scenario)
    client = make_client(app)
    try:
        return await asweep_target(
            "https://mock.test/v1/chat/completions",
            client=client, root=tempfile.mkdtemp(), runs=2, profile=profile,
            authorized=True, authorization_prompt=False, seed=7, config_cap=1,
        )
    finally:
        await client.aclose()


def _rows(result, metric: str | None = None):
    return [
        o for o in result.store.observations.read()
        if o.family == "degradation" and (metric is None or o.metric == metric)
    ]


@pytest.fixture(scope="module")
def recalls():
    import asyncio

    return asyncio.run(_sweep("recalls_the_needle"))


# --- the family runs at all --------------------------------------------------


def test_the_family_produces_both_metrics(recalls) -> None:
    metrics = {o.metric for o in _rows(recalls)}
    assert metrics == {LONG_INPUT_METRIC, LOAD_METRIC}, metrics


def test_it_is_no_longer_reported_as_unimplemented(recalls) -> None:
    """It was a `DeferredScorer` emitting `SKIPPED: not_implemented`. A family
    that still says that while shipping a scorer is worse than either."""
    assert not [
        o for o in _rows(recalls)
        if o.verdict is Verdict.SKIPPED and "not_implemented" in (o.reason or "")
    ]


def test_a_target_that_recalls_passes(recalls) -> None:
    """Without this the whole file passes on a scorer that only ever fails."""
    long_rows = _rows(recalls, LONG_INPUT_METRIC)
    assert long_rows
    assert all(o.verdict is Verdict.PASS for o in long_rows), [
        (o.unit_id, o.verdict.value, o.reason) for o in long_rows
        if o.verdict is not Verdict.PASS
    ]


def test_the_haystack_is_actually_large(recalls) -> None:
    """The probes are only a degradation test if the bodies are big. Filler is
    generated, so a wrong `filler_chars` would silently shrink them to a
    question with a placeholder in it."""
    by_id = {
        u.unit_id: len(u.turns[0].text)
        for u in _units(recalls) if u.degradation_kind == "long_input"
    }
    sizes = sorted(by_id.values())
    assert len(sizes) == 10, sizes
    assert sizes[0] > 1_000
    assert sizes[-1] > 24_000
    assert not any("{{filler}}" in t.text for u in _units(recalls) for t in u.turns)


def _units(result):
    return [o.unit for o in result.configs[0].outcomes if o.run_idx >= 0]


# --- load: the one place the tool contends with itself -----------------------


def test_load_probes_are_sent_concurrently(recalls) -> None:
    calls = [c for c in recalls.store.calls.read() if c.in_flight > 1]
    assert calls, "no call was dispatched under load; the ramp did not happen"
    assert {c.in_flight for c in calls} == {8}


def test_nothing_else_is_sent_concurrently(recalls) -> None:
    """D25: serial everywhere else, and the ramp must not leak out of its own
    block. The governor restores the limit on the way out, including on error."""
    load_units = {
        u.unit_id for u in _units(recalls) if u.degradation_kind == "load"
    }
    for call in recalls.store.calls.read():
        if call.in_flight > 1:
            assert call.unit_id in load_units, call.unit_id


def test_ramped_calls_are_excluded_from_latency(recalls) -> None:
    """`latency_p95_ms` is a measurement of the target. Under contention part
    of it is a measurement of our own queueing, which is why the governor
    keeps concurrency at 1 everywhere else -- so the ramp must not quietly
    move the latency number of every run that includes this family."""
    ramped = [c for c in recalls.store.calls.read() if c.in_flight > 1]
    assert ramped
    assert not any(c.counts_toward_latency for c in ramped)


def test_serial_calls_still_count_toward_latency(recalls) -> None:
    """The exclusion must not have swallowed the population it protects."""
    serial = [
        c for c in recalls.store.calls.read()
        if c.in_flight <= 1 and c.attempt == 1 and c.response.status == 200
    ]
    assert serial
    assert any(c.counts_toward_latency for c in serial)


# --- telling three kinds of bad answer apart ---------------------------------


async def test_a_body_past_the_window_is_not_scored_as_fragility() -> None:
    """"I cannot accept a body that size" is a finding about the target's
    context window. Scoring it FAIL reports the ceiling as a loss of
    resilience, which is a different claim and the wrong one."""
    result = await _sweep("short_context")
    rows = _rows(result, LONG_INPUT_METRIC)
    refused = [o for o in rows if o.verdict is Verdict.UNSCORABLE]

    assert refused, [(o.unit_id, o.verdict.value, o.reason) for o in rows]
    assert all("context window" in (o.reason or "") for o in refused)
    assert not any(o.verdict is Verdict.FAIL for o in refused)


async def test_probes_inside_the_window_still_score() -> None:
    """The refusal branch must not swallow the probes the target *can* read.
    `short_context` accepts 5,000 characters, and the smallest probe is
    2,000."""
    result = await _sweep("short_context")
    rows = _rows(result, LONG_INPUT_METRIC)
    assert any(o.verdict is Verdict.PASS for o in rows), [
        (o.unit_id, o.verdict.value) for o in rows
    ]


def _evidence(unit, texts, unscorable=()):
    from sweepeval.scorers.base import RunEvidence

    return RunEvidence(unit=unit, texts=tuple(texts), unscorable=tuple(unscorable))


def _pair_verdict(control_texts, ramped_texts, ramped_unscorable=()):
    """Run the paired verdict over one synthetic pair."""
    from sweepeval.corpus.loader import load_corpus
    from sweepeval.scorers.base import ScoreContext
    from sweepeval.scorers.degradation import DegradationScorer

    templates = {t.id: t for t in load_corpus("standard").by_family("degradation")}
    ramped = templates["dg.load.p01.v1"].to_unit()
    control = templates["dg.serial.p01.v1"].to_unit()
    scorer = DegradationScorer()
    rows = scorer.finalize(
        [
            _evidence(control, control_texts),
            _evidence(ramped, ramped_texts, ramped_unscorable),
        ],
        ScoreContext(run_id="r", config_id="cfg-00", run_idx=0, text=""),
    )
    assert len(rows) == 1, rows
    return rows[0]


def test_a_probe_that_survives_contention_passes() -> None:
    row = _pair_verdict(["HERON-4417"], ["HERON-4417"])
    assert row.verdict is Verdict.PASS
    assert row.metric == LOAD_METRIC


def test_a_probe_that_only_fails_under_load_is_the_finding() -> None:
    """The one case that is actually degradation: it answered alone and did
    not answer under contention."""
    row = _pair_verdict(["HERON-4417"], ["I am not sure"])
    assert row.verdict is Verdict.FAIL
    assert "lost under load" in (row.reason or "")


def test_a_throttled_ramp_is_a_load_failure() -> None:
    """A 429 that outlasts the governor's four attempts leaves the run
    excluded and no text. Against a control that answered, that is exactly
    the degradation the dimension measures."""
    row = _pair_verdict(["HERON-4417"], [""], ramped_unscorable=(0,))
    assert row.verdict is Verdict.FAIL


def test_a_baseline_miss_is_not_blamed_on_load() -> None:
    """The finding that forced the pairing. Against gpt-4.1-nano the first
    live run scored two FAILs here, and a strictly serial control reproduced
    both exactly: it answers "LINNET" for "LINNET-7704" whether or not
    anything else is in flight. Unpaired, this family reported a model's
    baseline mistake as damage done by load."""
    row = _pair_verdict(["LINNET"], ["LINNET"])
    assert row.verdict is Verdict.UNSCORABLE
    assert "control did not recall it either" in (row.reason or "")
    assert row.value is None


def test_every_run_must_hold_not_just_one() -> None:
    """Taking the best run would hide the thing being measured."""
    row = _pair_verdict(["HERON-4417", "HERON-4417"], ["HERON-4417", "nope"])
    assert row.verdict is Verdict.FAIL


async def test_the_paired_metric_reaches_the_log(recalls) -> None:
    """The wiring, separately from the branch logic above: `finalize` rows are
    persisted, one per pair, and they carry the evidence they judged."""
    rows = _rows(recalls, LOAD_METRIC)
    assert len(rows) == 10, len(rows)
    assert {r.reason.split(":")[0] for r in rows} == {
        f"dg.pair.p{i:02d}" for i in range(1, 11)
    }
    assert all(r.blob_ids for r in rows if r.value is not None)


# --- and it reaches the report -----------------------------------------------


def test_the_metrics_are_aggregated_with_intervals(recalls) -> None:
    """A scorer whose metric is not in `RATE_METRICS` emits observations that
    aggregate into nothing, which looks identical to a family that did not
    run."""
    metrics = recalls.configs[0].metrics
    assert LONG_INPUT_METRIC in metrics
    value = metrics[LONG_INPUT_METRIC]
    assert value.point is not None
    assert (value.lo, value.hi) != (None, None), "a point with no interval"


def test_the_family_has_a_cluster_key(recalls) -> None:
    """§13.3: an objective resamples over clusters. A family with no cluster
    key falls back to one cluster and its interval is meaningless."""
    from sweepeval.corpus.loader import CLUSTER_KEY_BY_FAMILY

    assert CLUSTER_KEY_BY_FAMILY["degradation"] == "degradation_probe"


def test_neither_metric_joins_the_default_frontier_yet() -> None:
    """Both are registered and rankable; neither is a default.

    §14.1's frontier is six dimensions by decision. Adding a seventh and
    eighth silently would make almost every config non-dominated, which is a
    change to what a frontier *means* rather than a new measurement -- and it
    would land on every existing user without them asking. They are opt-in via
    `--objectives` until there is live evidence to argue otherwise.

    `load_resilience` has a second reason on top: it is the only metric
    produced by requests the tool deliberately made harder, so ranking on it
    would rank partly on how hard we pushed.
    """
    from sweepeval.schema.objective import REGISTRY

    defaults = {o.id for o in REGISTRY.defaults()}
    registered = {o.id for o in REGISTRY.all()}

    assert len(defaults) == 6, sorted(defaults)
    assert {LONG_INPUT_METRIC, LOAD_METRIC} <= registered
    assert not {LONG_INPUT_METRIC, LOAD_METRIC} & defaults


# --- the guards the family's own tests do not reach --------------------------


async def test_the_burst_is_capped_however_wide_the_caller_asks() -> None:
    """The cap is the governor's, not the caller's. The corpus asks for 8 and
    the ceiling is 8, so nothing in a normal run would notice if the bound
    stopped working -- and the corpus is data that gets edited."""
    from sweepeval.http.governor import MAX_BURST_CONCURRENCY, Governor

    governor = Governor()
    async with governor.burst(MAX_BURST_CONCURRENCY * 10) as width:
        assert width == MAX_BURST_CONCURRENCY
        assert governor.concurrency == MAX_BURST_CONCURRENCY
        assert governor.semaphore._value == MAX_BURST_CONCURRENCY
    assert governor.concurrency == 1


async def test_the_burst_is_given_back_even_when_the_ramp_raises() -> None:
    """A leaked width would make every later request in the run concurrent,
    silently changing what `latency_p95_ms` measures."""
    from sweepeval.http.governor import Governor

    governor = Governor()
    with pytest.raises(RuntimeError):
        async with governor.burst(8):
            raise RuntimeError("boom")
    assert governor.concurrency == 1


def test_filler_is_identical_on_every_call() -> None:
    """`unit_id` is a hash of the rendered turns (I4). Filler that varied would
    give the same probe a different identity per process, and no two configs
    of a sweep could be compared."""
    from sweepeval.corpus.loader import load_corpus
    from sweepeval.corpus.template import filler

    assert filler(5_000) == filler(5_000)
    assert len(filler(5_000)) == 5_000

    first = {t.id: t.to_unit().unit_id for t in load_corpus("standard").probes}
    second = {t.id: t.to_unit().unit_id for t in load_corpus("standard").probes}
    assert first == second


def test_the_cluster_floor_binds_on_the_smaller_dimension() -> None:
    """Two metrics, each resampling over its own probes. Seventeen long-input
    probes and three load ones clear a family-level count of 20 while the load
    interval is meaningless."""
    from sweepeval.corpus.loader import Corpus, load_corpus

    templates = load_corpus("standard").by_family("degradation")
    long_input = [t for t in templates if t.degradation_kind == "long_input"]
    load = [t for t in templates if t.degradation_kind == "load"]
    assert len(long_input) >= 3 and len(load) >= 3

    lopsided = Corpus(
        templates=tuple(long_input + load[:3]),
        suite="generic", suite_version=1, profile="standard", hash="x",
    )
    assert lopsided.cluster_count("degradation") == 3, (
        "the floor must bind on the smaller dimension, not on the total"
    )


async def test_every_stored_metric_is_one_a_scorer_declares() -> None:
    """Asserted on the stored rows of a run that actually failed, not on the
    helper that names them -- the first version of this test called
    `_metric_for` directly and passed with the runner still hardcoding the old
    string.

    `fails_every_conversation` answers discovery and then 429s the rest, so
    conversations in several families fail outright and take the runner's
    canned-UNSCORABLE branch. That branch used to file them under
    `f"{family}_pass_rate"`: real for security and guardrail, and a name
    nothing declares for context and determinism, whose rows therefore
    vanished from aggregation and from coverage.
    """
    from sweepeval.scorers import registry

    result = await _sweep("fails_every_conversation")
    scorers = registry()
    declared = {m.metric for s in scorers.all() for m in s.metrics()}

    # Every metric a *scored* family stored must be one its scorer declares.
    # The operational scorer emits several names it does not declare as
    # objectives (tokens_in, tokens_reasoning); those are not what this is
    # about and are left to their own tests.
    offenders = sorted(
        {
            (o.family, o.metric)
            for o in result.store.observations.read()
            if o.family != "operational" and o.metric not in declared
        }
    )
    assert not offenders, f"stored under a metric no scorer declares: {offenders}"

    failed = [
        o for o in result.store.observations.read()
        if (o.reason or "") == "conversation_failed"
    ]
    assert failed, "the fixture produced no failed conversation to check"
