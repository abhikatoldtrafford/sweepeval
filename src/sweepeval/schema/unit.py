"""``Unit`` — the join key (spec §6.1, §10.1, §11.7).

A Unit is one instantiated, fully-resolved probe: a template with slots filled,
canary *names* attached, and a fixed turn script. It is the atom of planning,
execution and scoring, and the join key across ``plan.json``, ``calls.jsonl``
and ``observations.jsonl``.

**I4 load-bearing.** "Within a sweep every config faces the identical probe
set" is checkable rather than aspirational only because Units serialise into
``plan.json`` in full. Two properties follow, and both are enforced here:

* ``unit_id`` is stable across runs and processes, so observations from
  different runs join;
* the serialised form is canonical, so ``plan.json`` hashes identically in
  every process — ``--resume`` refuses on a plan-hash mismatch (§12.5), and a
  serialisation that depended on set or dict iteration order would make every
  resume fail for no real reason.

**Canaries.** A Unit carries ``canary_names``, never values. Values are derived
per ``run_idx`` (§11.2), and a Unit is run-independent by construction, so it
cannot hold them; they live in ``plan.json``'s ``canary_table``. Names are
excluded from ``unit_id`` so the id stays stable across runs.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from sweepeval.schema.hashing import param_hash

__all__ = ["OnRefusal", "Role", "ScoringContract", "Turn", "Unit"]

Role = Literal["system", "user", "assistant"]

OnRefusal = Literal["pass", "fail", "unscorable", "expected"]
"""§11.8's per-template override of the family default refusal policy.

It lives on the Unit rather than only on the template because the runner and
the scorers are what act on it, and they see Units. An earlier revision
declared it on all 62 templates and dropped it in ``to_unit()``, so no code
could read it -- which left a target that refuses *everything* scoring
``target_determinism_at_temp0 = 1.0``: three identical refusals are perfectly
repeatable.
"""

ContractKind = Literal[
    "canary_absent",
    "marker_present",
    "refusal_expected",
    "compliance_expected",
    "equivalence",
    "fact_recall",
]


class Turn(BaseModel):
    """One scripted turn. Never adaptive — an adaptive turn would unfreeze the
    probe set and violate I4 (§11.7)."""

    model_config = ConfigDict(frozen=True)

    role: Role
    text: str


class ScoringContract(BaseModel):
    """One deterministic assertion (§10.1).

    ``ambiguous_when`` names the condition under which the contract returns
    ``AMBIGUOUS`` rather than a verdict. Judge escalation (§11.9) triggers on
    that and nothing else, so a contract with no ``ambiguous_when`` can never
    reach the judge.
    """

    model_config = ConfigDict(frozen=True)

    kind: ContractKind
    canary: str | None = None
    marker: str | None = None
    expect: str | None = None
    ambiguous_when: str | None = None


class Unit(BaseModel):
    """One instantiated, fully-resolved probe."""

    model_config = ConfigDict(frozen=True)

    unit_id: str
    template_id: str
    family: str
    turns: tuple[Turn, ...]
    canary_names: tuple[str, ...] = ()
    scoring: tuple[ScoringContract, ...] = ()
    calls_per_run: int
    profiles: frozenset[str]
    params: dict[str, Any] = Field(default_factory=dict)
    severity: str | None = None
    attack_class: str | None = None
    policy_id: str | None = None
    depth: int | None = None
    expects_sources: bool | None = None
    """Whether this probe should surface retrieved sources; ``False`` means
    nothing could source it and citing anything is fabrication."""

    expects_tool: str | None = None
    """Which offered tool this probe should elicit; ``""`` means none should be.

    Excluded from ``unit_id`` like the rest of the scoring instructions: it
    says how the reply is judged, not what was sent.
    """

    group: str | None = None
    """Templates sharing one are two views of the same probe.

    The degradation family pairs each load probe with an identical serial
    control through it, so `load_resilience` can be a difference rather than
    an outcome. Excluded from ``unit_id``, like `on_refusal`: it says how the
    row is interpreted, not what was sent.
    """

    degradation_kind: str | None = None
    """``long_input``, ``load`` or ``serial_control`` for the degradation family (§11, family 8).

    The runner reads it to decide *how* to dispatch: ``load`` units are the
    only ones in the tool sent concurrently, and they have to be identifiable
    before scoring rather than after. Excluded from ``unit_id`` for the same
    reason ``on_refusal`` is -- it governs dispatch and scoring, not what was
    sent, and changing it must not break the join to an existing run's rows.
    """

    on_refusal: OnRefusal | None = None
    """None means "the family default" (§11.8). Excluded from ``unit_id``:
    it is a scoring instruction, not part of what was sent, so changing it
    must not break the join between an existing run's rows."""

    @field_validator("turns")
    @classmethod
    def _turns_nonempty(cls, value: tuple[Turn, ...]) -> tuple[Turn, ...]:
        if not value:
            raise ValueError("a Unit must have at least one turn")
        return value

    @field_serializer("profiles")
    def _serialise_profiles(self, value: frozenset[str]) -> list[str]:
        """Sort on the way out.

        Set iteration order for strings depends on PYTHONHASHSEED, so an
        unsorted serialisation makes plan.json's bytes differ between the
        process that wrote it and the process that resumes it.
        """
        return sorted(value)

    @field_serializer("params")
    def _serialise_params(self, value: dict[str, Any]) -> dict[str, Any]:
        """Sort keys, for the same reason.

        ``unit_id`` is already order-independent (``param_hash`` canonicalises),
        but the serialised bytes must be too, or two equal Units produce two
        different plan.json digests.
        """
        return dict(sorted(value.items()))

    def verify_id(self) -> bool:
        """Re-derive ``unit_id`` from the current ``params`` and compare.

        ``frozen=True`` blocks attribute assignment but does not deep-freeze
        the ``params`` dict, so ``unit.params["depth"] = 8`` succeeds and
        silently desynchronises params from the id it was derived from. The
        plan-freeze check (Task 8.1) calls this so the desync fails loudly
        instead of corrupting every join that uses ``unit_id``.
        """
        return self.unit_id == f"{self.template_id}#{param_hash(self.params)}"

    @classmethod
    def make(
        cls,
        *,
        template_id: str,
        family: str,
        turns: list[Turn],
        scoring: list[ScoringContract] | None = None,
        profiles: set[str],
        params: dict[str, Any] | None = None,
        canary_names: tuple[str, ...] = (),
        severity: str | None = None,
        attack_class: str | None = None,
        policy_id: str | None = None,
        depth: int | None = None,
        group: str | None = None,
        expects_tool: str | None = None,
        expects_sources: bool | None = None,
        degradation_kind: str | None = None,
        on_refusal: OnRefusal | None = None,
    ) -> Unit:
        """The only constructor. Derives ``unit_id`` and ``calls_per_run``.

        ``calls_per_run`` is deliberately not a parameter: it drives the budget
        estimate the user consents to under I9, and a hand-written value is how
        an earlier revision of the spec came to under-count every multi-turn
        unit (§10.1).
        """
        if not turns:
            raise ValueError("a Unit must have at least one turn")
        resolved = dict(params or {})
        return cls(
            # canary_names deliberately excluded: unit_id is stable across runs (§6.1)
            unit_id=f"{template_id}#{param_hash(resolved)}",
            template_id=template_id,
            family=family,
            turns=tuple(turns),
            canary_names=tuple(canary_names),
            scoring=tuple(scoring or ()),
            calls_per_run=len(turns),
            profiles=frozenset(profiles),
            params=resolved,
            severity=severity,
            attack_class=attack_class,
            policy_id=policy_id,
            depth=depth,
            group=group,
            expects_tool=expects_tool,
            expects_sources=expects_sources,
            degradation_kind=degradation_kind,
            on_refusal=on_refusal,
        )
