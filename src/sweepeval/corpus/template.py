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

from sweepeval.schema.unit import OnRefusal, ScoringContract, Turn, Unit

__all__ = [
    "FILLER_MAX_CHARS",
    "PROFILES",
    "OnRefusal",
    "ProbeTemplate",
    "Profile",
    "ScoringSpec",
    "TurnSpec",
    "filler",
]

Profile = Literal["quick", "standard", "deep"]
PROFILES: tuple[Profile, ...] = ("quick", "standard", "deep")

Family = Literal[
    "security", "guardrail", "determinism", "context", "operational",
    "degradation", "tool_integrity", "retrieval",
]

DegradationKind = Literal["long_input", "load", "serial_control"]

FILLER_MAX_CHARS = 200_000
"""Ceiling on generated filler, so a corpus file cannot author a huge request.

The corpus is data and is reviewed as data; a template that asked for ten
million characters would be a denial-of-service payload written in YAML and
waved through as a number. 200k characters is roughly 50k tokens, past the
context window of most of what this tool is pointed at, which is far enough
for the family's purpose.
"""

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

    degradation_kind: DegradationKind | None = None
    """Which degradation dimension this probe belongs to (§11, family 8).

    ``long_input`` probes are dispatched like any other. ``load`` probes are
    dispatched *together*, inside one concurrency burst, because contention is
    the condition under test and it cannot be produced one request at a time.
    """

    expects_sources: bool | None = None
    """Whether this probe should surface retrieved sources (§11, family 6).

    ``False`` is the interesting half: a question nothing could have a source
    for. A target that cites something anyway has fabricated it, which is the
    failure mode that matters most and is invisible in a corpus where every
    probe expects citations.
    """

    expects_tool: str | None = None
    """Which offered tool this probe should elicit (§11, family 4).

    ``""`` -- the empty string -- means the opposite and is not the same as
    absent: *no* tool should be called, because the question needs none. Those
    probes are how the family measures over-calling, which is a real failure
    mode and invisible if every probe expects a call.
    """

    filler_chars: int | None = None
    """Expand ``{{filler}}`` to this many characters of neutral prose.

    Generated rather than stored. The alternative is committing tens of
    kilobytes of lorem per probe, which bloats a corpus that exists to be read
    and diffed by people. Deterministic, so the rendered text -- and therefore
    the ``unit_id`` hashed from it (I4) -- is identical on every machine and
    every run.
    """

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
        if self.family == "retrieval" and self.expects_sources is None:
            raise ValueError(
                f"{self.id}: retrieval templates declare expects_sources"
            )
        if self.family == "tool_integrity" and self.expects_tool is None:
            raise ValueError(
                f"{self.id}: tool_integrity templates declare expects_tool "
                "(a tool name, or '' for 'no tool should be called')"
            )
        if self.expects_tool:
            from sweepeval.tools import BY_NAME

            if self.expects_tool not in BY_NAME:
                raise ValueError(
                    f"{self.id}: expects_tool {self.expects_tool!r} is not in the "
                    f"offered toolkit ({', '.join(sorted(BY_NAME))}); a probe "
                    "cannot expect a tool the target is never given"
                )
        if self.family == "degradation" and self.degradation_kind is None:
            raise ValueError(
                f"{self.id}: degradation templates declare a degradation_kind"
            )
        if self.degradation_kind is not None and self.family != "degradation":
            raise ValueError(
                f"{self.id}: degradation_kind belongs to the degradation family"
            )
        if self.filler_chars is not None:
            if self.filler_chars < 0:
                raise ValueError(f"{self.id}: filler_chars cannot be negative")
            if self.filler_chars > FILLER_MAX_CHARS:
                raise ValueError(
                    f"{self.id}: filler_chars {self.filler_chars} exceeds the "
                    f"{FILLER_MAX_CHARS} ceiling"
                )
        if "{{filler}}" in "".join(t.text for t in self.turns) and not self.filler_chars:
            raise ValueError(
                f"{self.id}: uses {{{{filler}}}} but declares no filler_chars, so "
                "the probe would be sent with the placeholder left in it"
            )
        return self

    @property
    def calls_per_run(self) -> int:
        """Derived from turns, never hand-written (§10.1).

        This is what the budget estimate the user consents to is built from,
        and a hand-written value is how rev 1 came to under-count every
        multi-turn unit.

        Only **user** turns are calls. A system turn is context carried into
        the next request, not a request of its own, so counting it would
        over-state the estimate by one call for every framed probe -- and the
        estimate is the number the user says yes to.
        """
        return sum(1 for turn in self.turns if turn.role == "user")

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
            Turn(
                role=t.role,
                text=_render(t.text, canaries or {}, self.slots, self.filler_chars),
            )
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
            group=self.group,
            expects_tool=self.expects_tool,
            expects_sources=self.expects_sources,
            degradation_kind=self.degradation_kind,
            on_refusal=self.on_refusal,
        )


def _render(
    text: str,
    canaries: dict[str, str],
    slots: dict[str, Any],
    filler_chars: int | None = None,
) -> str:
    out = text
    for name, value in canaries.items():
        out = out.replace(f"{{{{canary:{name}}}}}", value)
        if name == "primary":
            out = out.replace("{{canary}}", value)
    for name, value in slots.items():
        out = out.replace(f"{{{{{name}}}}}", str(value))
    if filler_chars:
        out = out.replace("{{filler}}", filler(filler_chars))
    return out


_FILLER_LINES = (
    "The depot logged routine inbound movements against the standing schedule.",
    "Pallet counts were reconciled against the manifest before dispatch.",
    "Cold storage held within the agreed range for the whole reporting period.",
    "Two loading bays were resurfaced, which shifted timings by a few minutes.",
    "Seasonal volumes tracked the forecast closely enough to need no revision.",
    "Driver rotas were unchanged and no overtime was booked against the depot.",
    "Packaging stock was replenished on the usual cycle from the regional hub.",
    "Returns were processed on arrival and folded into the weekly stock count.",
)
"""Neutral, dull, and deliberately not memorable.

Filler that is interesting competes with the needle for attention, so the
probe would measure salience rather than retrieval. It must also be
unobjectionable: filler that tripped a safety filter would turn a recall
measurement into a refusal measurement, and the refusal policy (§11.8) would
then exclude the trial that was supposed to be the test.
"""


def filler(chars: int) -> str:
    """``chars`` characters of neutral prose, identical everywhere.

    Numbered lines rather than one sentence repeated: a long run of identical
    text is unusually compressible and is not what a real long input looks
    like. The numbering also makes a truncated request visible in the stored
    body, which is worth more than the tokens it costs.

    Deterministic with no seed and no randomness, because the ``unit_id`` is a
    hash of the rendered turns (I4). Filler that varied would give the same
    probe a different identity per process, and no two configs of a sweep
    could be compared.
    """
    if chars <= 0:
        return ""
    out: list[str] = []
    total = 0
    index = 0
    while total < chars:
        line = f"{index:05d}. {_FILLER_LINES[index % len(_FILLER_LINES)]}\n"
        out.append(line)
        total += len(line)
        index += 1
    return "".join(out)[:chars]
