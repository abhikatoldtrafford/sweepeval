"""Finding the ambiguous observations and resolving them (spec §11.9).

The trigger is exact: an observation whose reason begins ``ambiguous:``, on a
unit whose contract declares a matching ``ambiguous_when``. Both halves are
required. A scorer that emitted the marker without the corpus declaring the
condition would be escalating on its own authority, and §11.9 says the
declared condition "and nothing else".

Resolution never overwrites the original row. The append-only log keeps the
deterministic UNSCORABLE and gains a second observation carrying the judge's
verdict, its confidence, its rationale and the prompt version -- so a reader
can see which numbers a model decided, and re-derive the metric without them
by filtering on scorer.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from sweepeval.judge.client import JudgeConfig, JudgeVerdict
from sweepeval.judge.rubric import rubric_for
from sweepeval.schema.observation import Observation, Verdict
from sweepeval.schema.unit import Unit

__all__ = [
    "AMBIGUOUS_PREFIX",
    "JUDGE_SCORER",
    "Escalation",
    "EscalationPlan",
    "plan_escalations",
    "resolved_observation",
]

AMBIGUOUS_PREFIX = "ambiguous:"
JUDGE_SCORER = "judge"


@dataclass(frozen=True)
class Escalation:
    """One ambiguity worth a judge call."""

    observation: Observation
    unit: Unit
    contract_kind: str
    condition: str

    @property
    def expectation(self) -> str:
        return f"the contract declares it ambiguous when: {self.condition}"

    def probe_text(self) -> str:
        """The user turns, which is what the target was actually asked."""
        return "\n\n".join(t.text for t in self.unit.turns if t.role == "user")


@dataclass
class EscalationPlan:
    """What would be escalated, before anything is sent."""

    escalations: list[Escalation] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    """``(unit_id, reason)`` for ambiguities the judge cannot take -- no
    rubric for the contract kind, or no stored response to judge. Recorded
    rather than dropped: an ambiguity nobody resolved and nobody mentioned is
    the silent gap this whole feature exists to close."""

    def __len__(self) -> int:
        return len(self.escalations)


def plan_escalations(
    observations: Iterable[Observation],
    units: Sequence[Unit],
    *,
    texts: dict[tuple[str, int], str] | None = None,
) -> EscalationPlan:
    """Which observations qualify. Sends nothing.

    Separated from execution so the pre-flight can count judge calls against
    the budget before the first one is made (§12.3, I9), and so a user can see
    what would be escalated without spending.
    """
    by_id = {unit.unit_id: unit for unit in units}
    plan = EscalationPlan()

    for observation in observations:
        if observation.verdict is not Verdict.UNSCORABLE:
            continue
        reason = observation.reason or ""
        if not reason.startswith(AMBIGUOUS_PREFIX):
            continue

        unit = by_id.get(observation.unit_id)
        if unit is None:
            plan.skipped.append((observation.unit_id, "unit not in the plan"))
            continue

        condition = reason[len(AMBIGUOUS_PREFIX) :].strip()
        contract = next(
            (
                c
                for c in unit.scoring
                if c.ambiguous_when and c.ambiguous_when == condition
            ),
            None,
        )
        if contract is None:
            # The scorer claimed an ambiguity the corpus does not declare.
            # §11.9 triggers on the declared condition and nothing else.
            plan.skipped.append(
                (
                    observation.unit_id,
                    f"no contract declares ambiguous_when={condition!r}",
                )
            )
            continue

        if rubric_for(contract.kind) is None:
            plan.skipped.append(
                (observation.unit_id, f"no rubric for contract kind {contract.kind!r}")
            )
            continue

        if texts is not None and not texts.get(
            (observation.unit_id, observation.run_idx)
        ):
            plan.skipped.append(
                (observation.unit_id, "no stored response text to judge")
            )
            continue

        plan.escalations.append(
            Escalation(
                observation=observation,
                unit=unit,
                contract_kind=contract.kind,
                condition=condition,
            )
        )

    return plan


def resolved_observation(
    escalation: Escalation, verdict: JudgeVerdict, judge: JudgeConfig
) -> Observation:
    """A second row carrying the judge's answer.

    The deterministic row stays. Two reasons: the append-only log must not be
    rewritten (I7), and a reader has to be able to tell a number a regex
    produced from one a model produced -- including by dropping the latter and
    recomputing, which filtering on ``scorer == "judge"`` allows.
    """
    original = escalation.observation
    return original.model_copy(
        update={
            "scorer": JUDGE_SCORER,
            "scorer_version": judge.prompt_version,
            "verdict": _as_verdict(verdict.verdict),
            "value": verdict.value,
            "reason": (
                f"judge({judge.model}@v{judge.prompt_version}) "
                f"{verdict.verdict.lower()} "
                f"confidence={verdict.confidence:.2f}: {verdict.rationale}"
            ),
        }
    )


def _as_verdict(name: str) -> Verdict:
    return {
        "PASS": Verdict.PASS,
        "FAIL": Verdict.FAIL,
        "UNSCORABLE": Verdict.UNSCORABLE,
    }[name]
