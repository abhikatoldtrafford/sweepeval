"""Degradation scorer (spec §11, family 8).

"Does this hold up when conditions get worse?" Two dimensions are measured and
the third says why it is not.

``degradation_resilience`` is recall with a fact buried in a growing haystack.
``load_resilience`` is the same recall question asked while the tool is
deliberately contending with itself. Both are `fact_recall` contracts, which
is the point: holding the question fixed and varying only the condition is
what makes the difference attributable to the condition.

**A request the target could not accept is not a failure of resilience.** A
body past the context window comes back as a terminal 400 naming the limit,
and scoring that FAIL would report the ceiling as fragility -- two different
findings, and the one the user can act on is "your window is smaller than this
probe". Those are UNSCORABLE with the reason, per I5.

Rate limiting under the ramp is treated the other way, and deliberately. A
429 that survives the governor's four attempts and its backoff is what a
client of that endpoint actually experiences under load, so it is a FAIL of
load resilience rather than an excuse. The distinction is: the long-input
dimension asks "can it read this", and being told no is an answer about
capacity; the load dimension asks "does it still work under contention", and
being throttled is the degradation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from sweepeval.capabilities.detect import Capability
from sweepeval.capabilities.normalise import normalise
from sweepeval.schema.call import Call, ErrorClass
from sweepeval.schema.metric import MetricSpec
from sweepeval.schema.observation import Observation, Verdict
from sweepeval.schema.unit import Unit
from sweepeval.scorers.base import RunEvidence, ScoreContext, register
from sweepeval.scorers.refusal import excludes_the_trial, looks_like_refusal

__all__ = ["LOAD_METRIC", "LONG_INPUT_METRIC", "DegradationScorer"]

LONG_INPUT_METRIC = "degradation_resilience"
LOAD_METRIC = "load_resilience"

_CONTEXT_LIMIT_MARKERS = (
    "context length",
    "context_length",
    "maximum context",
    "too many tokens",
    "reduce the length",
    "string too long",
    "input is too long",
    "request too large",
    "maximum prompt",
)
"""Phrases a provider uses to say "that body was too big for me".

Matched on the stored error text rather than on the status code: a 400 is also
what a malformed body gets, and the two must not be conflated -- one is a
finding about the target's window and the other is a bug in this tool.
"""


@dataclass
class DegradationScorer:
    family: str = "degradation"
    version: int = 1
    requires: frozenset[Capability] = field(default_factory=frozenset)
    """Nothing. That is the reason this family was built before the other two
    deferred ones: `tool_integrity` and `retrieval` need capabilities every
    endpoint measured so far reports UNSUPPORTED, so their first real evidence
    would come long after their code."""

    scores_failures: bool = True
    """This family scores conversations that never completed.

    Everywhere else a failed conversation is UNSCORABLE and that is right: it
    says nothing about the thing being measured. Here it is often the whole
    finding -- a 429 that outlasted the governor's backoff, or a body the
    target refused for its size -- so the runner hands those to `score`
    instead of filing a canned UNSCORABLE over them.
    """

    def metrics(self) -> list[MetricSpec]:
        return [
            MetricSpec(
                metric=LONG_INPUT_METRIC, family=self.family, direction="maximize",
                unit="rate", cluster_key="degradation_probe",
            ),
            MetricSpec(
                metric=LOAD_METRIC, family=self.family, direction="maximize",
                unit="rate", cluster_key="degradation_probe",
            ),
        ]

    def finalize(
        self, evidence: Sequence[RunEvidence], context: ScoreContext
    ) -> list[Observation]:
        """The paired load verdicts. See :func:`_paired`."""
        return _paired(self, evidence, context)

    def metric_for(self, unit: Unit) -> str:
        """Which of this family's two metrics a given probe belongs to.

        The runner asks this when a unit-run has to be recorded UNSCORABLE
        without reaching :meth:`score` -- a conversation that failed outright.
        A family with one metric never needs it; this one would otherwise file
        a throttled load probe under the long-input metric.
        """
        return LOAD_METRIC if unit.degradation_kind == "load" else LONG_INPUT_METRIC

    def score(
        self, unit: Unit, calls: Sequence[Call], context: ScoreContext
    ) -> list[Observation]:
        if unit.degradation_kind in ("load", "serial_control"):
            # Inputs to a paired verdict, not verdicts. `load_resilience` is
            # the difference between the two, and a difference cannot be
            # computed one unit-run at a time -- :meth:`finalize` gets them
            # all at once, exactly as the determinism family does.
            return []

        metric = self.metric_for(unit)
        emit = lambda **kw: [  # noqa: E731 - one shape, five call sites
            context.observation(
                scorer=self.family, version=self.version, metric=metric,
                family=self.family, unit=unit, **kw,
            )
        ]

        expected = next(
            (s.expect for s in unit.scoring if s.kind == "fact_recall" and s.expect),
            None,
        )
        if not expected:
            return emit(
                verdict=Verdict.SKIPPED,
                reason="template declares no expected fact",
            )

        refused_by_size = _context_limit(calls)
        if refused_by_size is not None:
            return emit(
                verdict=Verdict.UNSCORABLE,
                reason=(
                    f"the target refused a body of this size: {refused_by_size}. "
                    "That is its context window, not a loss of resilience"
                ),
            )

        answer = normalise(context.text)
        if not answer:
            if metric is LOAD_METRIC and _errored(calls):
                # Under contention this is the measurement, not a gap in it.
                return emit(
                    verdict=Verdict.FAIL, value=0.0,
                    reason=f"under load: {_errored(calls)}, no usable answer",
                )
            return emit(
                verdict=Verdict.UNSCORABLE,
                reason="no text extracted from the final turn",
            )

        recalled = normalise(expected) in answer
        if not recalled and excludes_the_trial(unit) and looks_like_refusal(
            context.text
        ):
            # §11.8: a declined probe says nothing about resilience.
            return emit(
                verdict=Verdict.UNSCORABLE,
                reason="declined, trial excluded (§11.8)",
            )

        where = "under load" if metric is LOAD_METRIC else f"{_size(unit)} chars"
        return emit(
            verdict=Verdict.PASS if recalled else Verdict.FAIL,
            value=1.0 if recalled else 0.0,
            reason=f"{where}: {'recalled' if recalled else 'lost'}",
        )


PAIRED_REASONS = {
    "held": "recalled alone and under load",
    "degraded": "recalled alone, lost under load",
    "baseline": "the serial control did not recall it either, so nothing here "
                "is attributable to load",
    "missing": "no serial control for this probe",
}


def _size(unit: Unit) -> int:
    return sum(len(turn.text) for turn in unit.turns)


def _errored(calls: Sequence[Call]) -> str:
    """A short description of the final failure, or ``""`` if there was none."""
    if not calls:
        return "no call was made"
    last = calls[-1]
    if last.response.error_class is ErrorClass.ok:
        return ""
    return f"HTTP {last.response.status or '-'} ({last.response.error_class.value})"


def _context_limit(calls: Sequence[Call]) -> str | None:
    """Whether the target refused the body for being too large.

    Read off the *stored* error text, which is why the store keeps error
    bodies: a status code alone cannot tell "your prompt is longer than my
    window" from "your JSON is wrong", and only the first is a finding about
    the target rather than about sweepeval.
    """
    for call in calls:
        if call.response.error_class is ErrorClass.ok:
            continue
        text = (call.response.error_excerpt or "").lower()
        for marker in _CONTEXT_LIMIT_MARKERS:
            if marker in text:
                return f"HTTP {call.response.status or '-'}, {marker!r}"
    return None


register(DegradationScorer())


def _paired(
    scorer: DegradationScorer,
    evidence: Sequence[RunEvidence],
    context: ScoreContext,
) -> list[Observation]:
    """`load_resilience`, as a difference rather than an outcome.

    A ramped probe that failed tells you nothing on its own: the first live
    run of this family scored two FAILs that a strictly serial control
    reproduced exactly. So each `dg.load.pNN` is paired by ``group`` with an
    identical `dg.serial.pNN`, and the verdict is about the *pair*:

    * the control recalled it and the ramped run did not  -> FAIL, and that is
      degradation under load
    * both recalled it                                    -> PASS
    * the control did not recall it either                -> UNSCORABLE, with
      the reason. The probe is beyond this target whatever the concurrency,
      and calling that a load failure would be the confusion this pairing
      exists to remove.
    """
    by_group: dict[str, dict[str, RunEvidence]] = {}
    for item in evidence:
        kind = item.unit.degradation_kind
        if kind in ("load", "serial_control") and item.unit.group:
            by_group.setdefault(item.unit.group, {})[kind] = item

    observations: list[Observation] = []
    for group, pair in sorted(by_group.items()):
        ramped = pair.get("load")
        if ramped is None:
            continue
        control = pair.get("serial_control")
        expected = next(
            (
                s.expect
                for s in ramped.unit.scoring
                if s.kind == "fact_recall" and s.expect
            ),
            None,
        )
        if control is None or not expected:
            observations.append(
                _observe(scorer, context, ramped, Verdict.UNSCORABLE,
                         PAIRED_REASONS["missing"], group)
            )
            continue

        held = _recalled(control, expected)
        under = _recalled(ramped, expected)
        if not held:
            verdict, reason, value = (
                Verdict.UNSCORABLE, PAIRED_REASONS["baseline"], None
            )
        elif under:
            verdict, reason, value = Verdict.PASS, PAIRED_REASONS["held"], 1.0
        else:
            verdict, reason, value = Verdict.FAIL, PAIRED_REASONS["degraded"], 0.0
        observations.append(
            _observe(scorer, context, ramped, verdict, reason, group, value)
        )
    return observations


def _recalled(item: RunEvidence, expected: str) -> bool:
    """Whether every scorable run of this probe carried the fact.

    Every run, not any: a probe that answered once in three has degraded, and
    taking the best run would hide exactly what the family measures. Runs
    excluded under §11.8 -- refused, or the conversation failed -- do not
    count as recall, which is why a throttled ramp reads as a loss.
    """
    scorable = [
        text for i, text in enumerate(item.texts) if i not in item.unscorable
    ]
    if not scorable:
        return False
    needle = normalise(expected)
    return all(needle in normalise(text) for text in scorable)


def _observe(
    scorer: DegradationScorer,
    context: ScoreContext,
    item: RunEvidence,
    verdict: Verdict,
    reason: str,
    group: str,
    value: float | None = None,
) -> Observation:
    return context.observation(
        scorer=scorer.family, version=scorer.version, metric=LOAD_METRIC,
        family=scorer.family, verdict=verdict, value=value,
        reason=f"{group}: {reason}", unit=item.unit,
        blob_ids=tuple(b for b in item.blob_ids if b),
    )
