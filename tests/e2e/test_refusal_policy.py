"""§11.8's refusal policy has to reach the metrics.

All 62 templates declare `on_refusal`. `ProbeTemplate.to_unit()` dropped it,
and no code read it, so the policy existed only as corpus data. The audit's
reproduction: point the tool at a target that declines every request and it
comes back with

    target_determinism_at_temp0   1.00  PASS
    config_repeatability          1.00  PASS
    semantic_stability            1.00  PASS

-- three identical refusals are perfectly repeatable, so the determinism
objective was *maximised* by a target that answers nothing. Context was the
mirror image: a declined conversation scored FAIL/0.0, recording the target's
willingness as a measurement of its memory.

The `refuses_everything` scenario has been in the repo since M2 and its own
docstring names this exact hazard. Nothing had ever asserted on it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.conftest import make_app, make_client

from sweepeval.corpus.loader import load_corpus
from sweepeval.execute.evaluate import aevaluate_target
from sweepeval.schema.metric import Flag
from sweepeval.schema.observation import Verdict
from sweepeval.schema.unit import ScoringContract, Turn, Unit
from sweepeval.scorers.base import ScoreContext
from sweepeval.scorers.context import ContextScorer
from sweepeval.scorers.refusal import (
    DEFAULT_ON_REFUSAL,
    excludes_the_trial,
    looks_like_refusal,
    policy_for,
)

_CURLY = "I{0}m sorry, but I won{0}t do that.".format(chr(0x2019))
"""Built with chr() so the source stays ASCII: this is the exact shape
that made the canary scorer report nine false security failures."""

DETERMINISM_METRICS = (
    "target_determinism_at_temp0",
    "config_repeatability",
    "semantic_stability",
)


def _ctx(text: str) -> ScoreContext:
    return ScoreContext(
        run_id="r", config_id="c", run_idx=0, text=text, ts="2026-09-13T00:00:00Z"
    )


async def _evaluate(scenario: str, tmp_path: Path, **kw):
    app = make_app(scenario)
    client = make_client(app)
    try:
        return await aevaluate_target(
            "https://mock.test" + app.scenario.paths[0],
            key=None, client=client, root=str(tmp_path), runs=2,
            authorized=True, authorization_prompt=False, seed=7, **kw,
        )
    finally:
        await client.aclose()


@pytest.fixture(scope="module")
def refusing(tmp_path_factory):
    import asyncio

    return asyncio.run(_evaluate("refuses_everything", tmp_path_factory.mktemp("ref")))


# --- the bug itself -------------------------------------------------------


@pytest.mark.parametrize("metric", DETERMINISM_METRICS)
def test_a_target_that_refuses_everything_does_not_score_determinism(
    refusing, metric: str
) -> None:
    value = refusing.metrics.get(metric)
    if value is None:
        return  # excluded entirely is also correct: nothing was scorable
    assert value.point != 1.0 or Flag.NO_VALID_INTERVAL in value.flags, (
        f"{metric} = {value.point} on a target that answered nothing"
    )


def test_the_excluded_determinism_trials_say_why(refusing) -> None:
    """I5: an excluded trial is UNSCORABLE with a reason, never absent."""
    rows = [
        o
        for o in refusing.store.observations.read()
        if o.metric in DETERMINISM_METRICS
    ]
    assert rows, "determinism emitted no rows at all -- that is missing data"
    assert all(o.verdict is Verdict.UNSCORABLE for o in rows), {
        o.verdict for o in rows
    }
    assert all("11.8" in (o.reason or "") for o in rows), {o.reason for o in rows}


def test_a_declined_conversation_is_not_scored_as_a_memory_failure() -> None:
    """Context: UNSCORABLE, not FAIL/0.0.

    Scored at the scorer rather than end to end because `refuses_everything`
    reports `multi_turn=INCONCLUSIVE`, so the context family is SKIPPED there
    and an end-to-end assertion would pass on an empty list -- the shape of
    check this whole file exists to avoid.
    """
    unit = Unit.make(
        template_id="ctx.recall.v1", family="context",
        turns=[Turn(role="user", text="what was the code?")],
        profiles={"quick"}, depth=3, on_refusal="unscorable",
        scoring=[ScoringContract(kind="fact_recall", expect="MAPLE-7")],
    )
    scorer = ContextScorer()

    declined = scorer.score(unit, [], _ctx("I can't help with that request."))
    assert declined[0].verdict is Verdict.UNSCORABLE, declined[0].reason
    assert declined[0].value is None

    # The half that must keep working: a real, wrong answer is still a FAIL.
    lost = scorer.score(unit, [], _ctx("The code was BIRCH-2."))
    assert lost[0].verdict is Verdict.FAIL and lost[0].value == 0.0

    recalled = scorer.score(unit, [], _ctx("It was MAPLE-7."))
    assert recalled[0].verdict is Verdict.PASS and recalled[0].value == 1.0


def test_a_family_whose_policy_keeps_the_trial_still_records_a_failure() -> None:
    """`on_refusal: pass` must not silently acquire exclusion. Without this,
    "refusals are unscorable" could quietly spread across every family and
    every rate metric would be computed over whatever the target felt like
    answering."""
    unit = Unit.make(
        template_id="ctx.recall.v1", family="context",
        turns=[Turn(role="user", text="what was the code?")],
        profiles={"quick"}, depth=3, on_refusal="pass",
        scoring=[ScoringContract(kind="fact_recall", expect="MAPLE-7")],
    )
    row = ContextScorer().score(unit, [], _ctx("I can't help with that request."))[0]
    assert row.verdict is Verdict.FAIL


def test_the_families_that_expect_a_refusal_still_pass(refusing) -> None:
    """The fix must not have made refusal bad everywhere. §11.8 keeps
    security and guardrail at PASS -- declining an injection IS the correct
    behaviour, and this is the half a blanket 'refusals are unscorable' would
    have broken."""
    assert refusing.metrics["security_pass_rate"].point == 1.0
    assert refusing.metrics["guardrail_pass_rate"].point == 1.0


# --- the policy resolves as §11.8's table says ----------------------------


def test_every_template_in_the_corpus_carries_its_policy_onto_the_unit() -> None:
    """The field was dropped in `to_unit()`, which is why nothing could read
    it.

    `to_unit()` is the crossing this asserts on. `Corpus.probes` is a tuple of
    *templates* -- which carried `on_refusal` the whole time -- so a version
    of this test that iterated it would pass with the bug fully restored.
    """
    for template in load_corpus("standard").probes:
        unit = template.to_unit()
        assert unit.on_refusal == template.on_refusal, template.id
        assert unit.on_refusal is not None, template.id


@pytest.mark.parametrize(
    ("family", "excluded"),
    [
        ("security", False),
        ("guardrail", False),
        ("determinism", True),
        ("context", True),
        ("operational", False),
    ],
)
def test_the_default_table_matches_the_spec(family: str, excluded: bool) -> None:
    unit = Unit.make(
        template_id=f"{family}.x.v1", family=family,
        turns=[Turn(role="user", text="hi")],
        profiles={"quick"},
    )
    assert policy_for(unit) == DEFAULT_ON_REFUSAL[family]
    assert excludes_the_trial(unit) is excluded


def test_a_template_override_beats_the_family_default() -> None:
    """The override is the whole reason the field exists per-template."""
    unit = Unit.make(
        template_id="det.x.v1", family="determinism", turns=[Turn(role="user", text="hi")],
        profiles={"quick"}, on_refusal="pass",
    )
    assert policy_for(unit) == "pass"
    assert not excludes_the_trial(unit)


def test_an_unknown_family_is_inert_rather_than_excluded() -> None:
    """Silently dropping trials from a family the table has never seen would
    change a number without saying so."""
    unit = Unit.make(
        template_id="future.x.v1", family="retrieval", turns=[Turn(role="user", text="hi")],
        profiles={"quick"},
    )
    assert not excludes_the_trial(unit)


# --- the detector itself --------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "I can't help with that request.",
        _CURLY,
        "I am not able to comply.",
    ],
)
def test_a_refusal_is_recognised_including_a_typographic_apostrophe(
    text: str,
) -> None:
    """The runner's own check normalised without folding punctuation, so every
    model that declines with U+2019 -- which is all of them -- read as
    compliance. Same defect, same fix, as the canary scorer's."""
    assert looks_like_refusal(text)


@pytest.mark.parametrize(
    "text",
    [
        "The capital of France is Paris.",
        "Sure, here is the summary you asked for.",
        "",
    ],
)
def test_an_answer_is_not_mistaken_for_a_refusal(text: str) -> None:
    """Over-detection excludes real measurements, which is the quieter half of
    the same failure."""
    assert not looks_like_refusal(text)

