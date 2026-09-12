"""Probe templates (spec §10.1, D3).

Data only. The generic suite and any later generated probes share one schema,
so the corpus hashes, diffs and reviews as data, and a community contribution
carries no execution risk.

Slots exist from day one and are unused by the generic suite. Phase 14's
generator then adds a generator over an existing format rather than forcing a
corpus migration — which would change the corpus hash and invalidate every
accumulated baseline.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sweepeval.schema.unit import ScoringContract, Turn, Unit

__all__ = ["PROFILES", "ProbeTemplate", "Profile", "ScoringSpec", "TurnSpec"]

Profile = Literal["quick", "standard", "deep"]
PROFILES: tuple[Profile, ...] = ("quick", "standard", "deep")

Family = Literal["security", "guardrail", "determinism", "context", "operational"]

OnRefusal = Literal["pass", "fail", "unscorable", "expected"]


class TurnSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["system", "user", "assistant"] = "user"
    text: str


class ScoringSpec(BaseModel):
    """One deterministic assertion (§10.1).

    ``ambiguous_when`` names the condition under which the contract declines to
    return a verdict. Judge escalation (§11.9) triggers on that and nothing
    else, so a contract without it can never reach the judge.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[
        "canary_absent",
        "marker_present",
        "refusal_expected",
        "compliance_expected",
        "equivalence",
        "fact_recall",
    ]
    canary: str | None = None
    marker: str | None = None
    expect: str | None = None
    ambiguous_when: str | None = None


class ProbeTemplate(BaseModel):
    """One probe, before slot resolution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    suite: str = "generic"
    suite_version: int = 1
    family: Family
    profiles: tuple[Profile, ...]
    turns: tuple[TurnSpec, ...]
    scoring: tuple[ScoringSpec, ...] = ()
    on_refusal: OnRefusal | None = None
    slots: dict[str, Any] = Field(default_factory=dict)

    # family-specific cells, all optional
    attack_class: str | None = None
    severity: Literal["low", "medium", "high", "critical"] | None = None
    owasp: str | None = None
    """OWASP LLM Top 10 mapping, e.g. ``LLM01``. Descriptive, not conformance."""

    policy_id: str | None = None
    pressure_level: int | None = None
    depth: int | None = None
    group: str | None = None
    """Invariance group: templates sharing one are paraphrases of each other."""

    cluster_key: str | None = None
    """Overrides the family default. See §13.3's cluster table."""

    @model_validator(mode="after")
    def _check(self) -> ProbeTemplate:
        if not self.turns:
            raise ValueError(f"{self.id}: a template needs at least one turn")
        if not self.profiles:
            raise ValueError(f"{self.id}: a template must belong to a profile")
        if self.family == "security" and not self.attack_class:
            raise ValueError(f"{self.id}: security templates declare an attack_class")
        if self.family == "guardrail" and not self.policy_id:
            raise ValueError(f"{self.id}: guardrail templates declare a policy_id")
        if self.family == "context" and self.depth is None:
            raise ValueError(f"{self.id}: context templates declare a depth")
        return self

    @property
    def calls_per_run(self) -> int:
        """Derived from turns, never hand-written (§10.1).

        This is what the budget estimate the user consents to is built from,
        and a hand-written value is how rev 1 came to under-count every
        multi-turn unit.
        """
        return len(self.turns)

    def canary_names(self) -> tuple[str, ...]:
        names = [s.canary for s in self.scoring if s.canary]
        return tuple(dict.fromkeys(names))

    def to_unit(self, canaries: dict[str, str] | None = None) -> Unit:
        """Instantiate. ``canaries`` renders ``{{canary}}`` placeholders only.

        Values never reach the Unit itself (§6.1): a Unit is run-independent,
        canary values are derived per ``run_idx``, and the mapping lives in
        ``plan.json``'s canary table.
        """
        rendered = [
            Turn(role=t.role, text=_render(t.text, canaries or {}, self.slots))
            for t in self.turns
        ]
        return Unit.make(
            template_id=self.id,
            family=self.family,
            turns=rendered,
            scoring=[
                ScoringContract(
                    kind=s.kind,
                    canary=s.canary,
                    marker=s.marker,
                    expect=s.expect,
                    ambiguous_when=s.ambiguous_when,
                )
                for s in self.scoring
            ],
            profiles=set(self.profiles),
            params=dict(self.slots),
            canary_names=self.canary_names(),
            severity=self.severity,
            attack_class=self.attack_class,
            policy_id=self.policy_id,
            depth=self.depth,
        )


def _render(text: str, canaries: dict[str, str], slots: dict[str, Any]) -> str:
    out = text
    for name, value in canaries.items():
        out = out.replace(f"{{{{canary:{name}}}}}", value)
        if name == "primary":
            out = out.replace("{{canary}}", value)
    for name, value in slots.items():
        out = out.replace(f"{{{{{name}}}}}", str(value))
    return out
