"""Tool integrity scorer (spec §11, family 4).

Two questions, kept apart because they fail independently:

``tool_call_validity``       did the target call the right tool, with
                             arguments the schema it was sent will accept
``tool_selection_stability`` does the same prompt pick the same tool twice

sweepeval offers the toolkit, so every judgement here rests on a schema we
wrote rather than on anything assumed about the target. An unknown tool name
is a hallucination, a missing required argument is an incomplete call, and an
argument of the wrong JSON type is drift -- all decidable, none of it
guesswork.

**Calling nothing is scored, and so is calling too much.** Three probes expect
*no* call, because a target that reaches for a tool on every turn is as broken
as one that never reaches for the right one, and it bills for the privilege.
The verdict is FAIL there, not UNSCORABLE: the target did something wrong, and
it is not an inability to measure.

**Imitating a tool call in prose is not tool calling, and not nothing.** A
model with no native support will happily write ``<tool_call>{...}`` into its
message. Scoring that as a valid call would credit a target for something it
cannot do; scoring it as silence would lose the most useful thing the run
found. It is UNSCORABLE with the encoding named, and the capability report
carries the same fact -- which is the only place the distinction is actionable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from sweepeval.capabilities.detect import Capability
from sweepeval.schema.call import Call
from sweepeval.schema.metric import MetricSpec
from sweepeval.schema.observation import Observation, Verdict
from sweepeval.schema.unit import Unit
from sweepeval.scorers.base import RunEvidence, ScoreContext, register
from sweepeval.tools import extract_tool_calls, validate_call

__all__ = ["SELECTION_METRIC", "VALIDITY_METRIC", "ToolIntegrityScorer"]

VALIDITY_METRIC = "tool_call_validity"
SELECTION_METRIC = "tool_selection_stability"

NO_TOOL = "(none)"
"""What a turn with no tool call contributes to the selection comparison.

A distinct value rather than a skip: "called nothing twice" is stable, and
"called nothing, then called something" is not. Dropping the empty case would
score an intermittent caller as perfectly consistent.
"""


@dataclass
class ToolIntegrityScorer:
    family: str = "tool_integrity"
    version: int = 1
    requires: frozenset[Capability] = field(
        default_factory=lambda: frozenset({Capability.TOOL_CALLING})
    )
    scores_without_text: bool = True
    """This family compares what was *called*, not what was said.

    A reply carrying only tool calls has `content: null`, so the extracted
    text is empty precisely when the probe worked. Without this the cross-run
    pass excluded every successful run and `tool_selection_stability` reported
    "fewer than two scorable runs" on every probe that called a tool -- found
    on the family's first live run against api.openai.com.
    """

    """Still gated on the capability -- but the detector now offers a tool
    before deciding, so this no longer means "every endpoint"."""

    def metrics(self) -> list[MetricSpec]:
        return [
            MetricSpec(
                metric=VALIDITY_METRIC, family=self.family, direction="maximize",
                unit="rate", cluster_key="tool_probe",
            ),
            MetricSpec(
                metric=SELECTION_METRIC, family=self.family, direction="maximize",
                unit="rate", cluster_key="tool_probe",
            ),
        ]

    def metric_for(self, unit: Unit) -> str:
        return VALIDITY_METRIC

    def score(
        self, unit: Unit, calls: Sequence[Call], context: ScoreContext
    ) -> list[Observation]:
        def emit(verdict: Verdict, reason: str, value: float | None = None):
            return [
                context.observation(
                    scorer=self.family, version=self.version,
                    metric=VALIDITY_METRIC, family=self.family,
                    verdict=verdict, value=value, reason=reason, unit=unit,
                )
            ]

        if unit.expects_tool is None:
            return emit(Verdict.SKIPPED, "template declares no expects_tool")

        emitted = [
            validate_call(c)
            for c in extract_tool_calls(context.payload, context.text)
        ]
        imitated = [c for c in emitted if c.encoding == "text.embedded"]
        structured = [c for c in emitted if c.encoding != "text.embedded"]

        if imitated and not structured:
            return emit(
                Verdict.UNSCORABLE,
                f"the target wrote a tool call into its message "
                f"({imitated[0].encoding}: {imitated[0].name}) rather than "
                "emitting one; that is imitation, not tool calling",
            )

        if not unit.expects_tool:
            if not structured:
                return emit(Verdict.PASS, "no tool was needed and none was called", 1.0)
            return emit(
                Verdict.FAIL,
                "no tool was needed and it called "
                f"{', '.join(sorted({c.name for c in structured}))}",
                0.0,
            )

        if not structured:
            return emit(
                Verdict.FAIL,
                f"expected a call to {unit.expects_tool} and none was emitted",
                0.0,
            )

        chosen = structured[0]
        if any("no tool named" in problem for problem in chosen.problems):
            # Reported ahead of the mismatch: "you called a tool that does not
            # exist" and "you called the wrong one of the three I offered" are
            # different defects, and the first is the more serious.
            return emit(
                Verdict.FAIL,
                f"called {chosen.name!r}, which was never offered "
                f"(expected {unit.expects_tool!r}); {'; '.join(chosen.problems)}",
                0.0,
            )
        if chosen.name != unit.expects_tool:
            return emit(
                Verdict.FAIL,
                f"called {chosen.name!r}, expected {unit.expects_tool!r}",
                0.0,
            )
        if not chosen.ok:
            return emit(
                Verdict.FAIL,
                f"called {chosen.name} but {'; '.join(chosen.problems)}",
                0.0,
            )
        return emit(
            Verdict.PASS, f"called {chosen.name} with arguments the schema accepts", 1.0
        )

    def finalize(
        self, evidence: Sequence[RunEvidence], context: ScoreContext
    ) -> list[Observation]:
        """``tool_selection_stability``: same prompt, same choice?

        Separate from validity because they fail independently. A target can
        pick the right tool every time and malform its arguments every time,
        or pick a different tool on each run and format all of them perfectly.
        Collapsing the two would hide whichever is working.
        """
        observations: list[Observation] = []
        for item in evidence:
            if item.unit.family != self.family or item.unit.expects_tool is None:
                continue
            if not item.payloads or all(p is None for p in item.payloads):
                observations.append(
                    _unscorable(
                        self, context, item,
                        "no parsed response was carried into the cross-run "
                        "comparison, so which tool each run chose is unknown "
                        "(a resumed run does not re-parse its stored bodies)",
                    )
                )
                continue

            chosen = [
                _selection(item.payloads[i] if i < len(item.payloads) else None,
                           item.texts[i] if i < len(item.texts) else "")
                for i in range(len(item.texts))
                if i not in item.unscorable
            ]
            if len(chosen) < 2:
                observations.append(
                    _unscorable(
                        self, context, item,
                        f"fewer than two scorable runs "
                        f"({len(item.unscorable)} excluded under §11.8)",
                    )
                )
                continue
            stable = len(set(chosen)) == 1
            observations.append(
                context.observation(
                    scorer=self.family, version=self.version,
                    metric=SELECTION_METRIC, family=self.family,
                    verdict=Verdict.PASS if stable else Verdict.FAIL,
                    value=1.0 if stable else 0.0,
                    reason=(
                        f"chose {chosen[0]} on every run"
                        if stable
                        else f"chose {' then '.join(chosen)}"
                    ),
                    unit=item.unit,
                    blob_ids=tuple(b for b in item.blob_ids if b),
                )
            )
        return observations


def _unscorable(
    scorer: ToolIntegrityScorer,
    context: ScoreContext,
    item: RunEvidence,
    reason: str,
) -> Observation:
    return context.observation(
        scorer=scorer.family, version=scorer.version, metric=SELECTION_METRIC,
        family=scorer.family, verdict=Verdict.UNSCORABLE, reason=reason,
        unit=item.unit, blob_ids=tuple(b for b in item.blob_ids if b),
    )


def _selection(payload: object, text: str) -> str:
    """Which tool one run chose.

    Read from the parsed response, because a target that emits only a call
    returns a null `content` -- the choice is not in the text at all. Text is
    still passed in so an imitated call written into prose is counted as the
    choice it announces: a target that flips between imitating and emitting is
    unstable, and reading only the structured field would call it consistent.
    """
    calls = extract_tool_calls(payload, text)
    if not calls:
        return NO_TOOL
    return f"{calls[0].encoding}:{calls[0].name}"


register(ToolIntegrityScorer())
