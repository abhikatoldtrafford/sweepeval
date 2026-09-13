"""``--prefer`` — answering the question (spec §14.4, D40). I1 load-bearing.

The frontier alone can end in "12 configs, 1 cluster, all statistically tied",
which is a refusal to answer rather than a refusal to oversimplify.

``--prefer`` applies the **user's** declared preference to the stored frontier
at report time. Two grammars:

``--prefer security,cost,latency``
    Lexicographic priority. Take the configs nothing beats on the first
    objective; if that leaves more than one, apply the second; and so on.
``--prefer "maximize security_pass_rate subject to latency_p95_ms < 2000"``
    Constrained optimisation, with the constraint tested against the
    interval's favourable bound like every other constraint (§14.1).

I1 holds exactly: nothing composite is computed, nothing composite is stored,
the weighting is the user's rather than the tool's, it is applied offline to
stored aggregates, and changing it needs no re-run. It is off by default.

"Beats" here means *statistically* beats — the same paired test the frontier
uses, not a point comparison. A preference resolved on point estimates would
pick the luckiest config rather than the best one, and at ``quick`` that is
usually a different config on every seed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from sweepeval.rank.frontier import FrontierResult
from sweepeval.schema.metric import Flag, MetricValue
from sweepeval.schema.objective import REGISTRY, Objective

__all__ = [
    "ALIASES",
    "Preference",
    "PreferenceResult",
    "apply_preference",
    "parse_preference",
]

ALIASES: dict[str, str] = {
    "security": "security_pass_rate",
    "guardrail": "guardrail_pass_rate",
    "guardrails": "guardrail_pass_rate",
    "determinism": "target_determinism_at_temp0",
    "repeatability": "config_repeatability",
    "retention": "context_retention_auc",
    "context": "context_retention_auc",
    "latency": "latency_p95_ms",
    "cost": "cost_per_probe",
    "tokens": "cost_per_probe",
}
"""Short names accepted on the command line. The canonical id is always what
gets printed back, so a user can see what their shorthand resolved to."""


@dataclass(frozen=True)
class Bound:
    metric: str
    op: str
    value: float

    def describe(self) -> str:
        return f"{self.metric} {self.op} {self.value:g}"

    def satisfied_by(self, value: MetricValue | None) -> bool:
        """Tested on the favourable bound, like every other constraint.

        An unmeasured metric does not satisfy a constraint on it: a config the
        user asked to be under 2000ms, whose latency was never measured, has
        not been shown to be under 2000ms.
        """
        if value is None or Flag.NO_VALID_INTERVAL in value.flags:
            return False
        if value.lo is None or value.hi is None:
            return False
        if self.op in ("<", "<="):
            return value.lo <= self.value if self.op == "<=" else value.lo < self.value
        return value.hi >= self.value if self.op == ">=" else value.hi > self.value


@dataclass(frozen=True)
class Preference:
    """A parsed preference. Never stored in an artifact as a ranking."""

    kind: str
    """``lexicographic`` or ``constrained``."""

    priority: tuple[str, ...] = ()
    optimise: str = ""
    direction: str = "maximize"
    subject_to: tuple[Bound, ...] = ()
    source: str = ""

    def describe(self) -> str:
        if self.kind == "lexicographic":
            return f"prefer {', '.join(self.priority)} (in that order)"
        bounds = " and ".join(b.describe() for b in self.subject_to)
        return f"{self.direction} {self.optimise}" + (
            f" subject to {bounds}" if bounds else ""
        )


@dataclass
class PreferenceResult:
    preference: Preference
    chosen: str | None = None
    concedes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """Objective -> configs that beat the chosen one on it."""

    excluded: dict[str, str] = field(default_factory=dict)
    """Config -> the bound it failed."""

    tie_broken: bool = False
    reason: str = ""


_CONSTRAINED = re.compile(
    r"^\s*(maximi[sz]e|minimi[sz]e)\s+([A-Za-z0-9_]+)"
    r"(?:\s+subject\s+to\s+(.+))?\s*$",
    re.I,
)
_BOUND = re.compile(r"([A-Za-z0-9_]+)\s*(<=|>=|<|>)\s*(-?[0-9.]+)")


def _metric_key(name: str) -> str:
    """The cluster-table key behind an objective id, when one is registered."""
    try:
        return REGISTRY.get(name).metric
    except KeyError:
        return name


def resolve(name: str) -> str:
    key = name.strip().lower()
    return ALIASES.get(key, key)


def parse_preference(text: str) -> Preference:
    """Parse either grammar, or raise with the two forms spelled out."""
    raw = text.strip()
    if not raw:
        raise ValueError("--prefer needs a preference")

    match = _CONSTRAINED.match(raw)
    if match:
        direction = "maximize" if match.group(1).lower().startswith("maxim") else "minimize"
        bounds = tuple(
            Bound(metric=resolve(m), op=op, value=float(v))
            for m, op, v in _BOUND.findall(match.group(3) or "")
        )
        if match.group(3) and not bounds:
            raise ValueError(
                f"could not read the constraint in {raw!r}; expected something "
                "like 'subject to latency_p95_ms < 2000'"
            )
        return Preference(
            kind="constrained",
            optimise=resolve(match.group(2)),
            direction=direction,
            subject_to=bounds,
            source=raw,
        )

    parts = [p for p in (piece.strip() for piece in raw.split(",")) if p]
    if not parts:
        raise ValueError(f"could not read a preference from {raw!r}")
    if any(" " in part for part in parts):
        raise ValueError(
            f"could not read {raw!r}. Two forms are accepted:\n"
            "  --prefer security,cost,latency\n"
            '  --prefer "maximize security_pass_rate subject to '
            'latency_p95_ms < 2000"'
        )
    return Preference(
        kind="lexicographic",
        priority=tuple(resolve(p) for p in parts),
        source=raw,
    )


def apply_preference(
    preference: Preference,
    result: FrontierResult,
    metrics: Mapping[str, Mapping[str, MetricValue]],
) -> PreferenceResult:
    """Name one config from the frontier, and say what it gives up."""
    out = PreferenceResult(preference=preference)
    candidates = list(result.frontier)

    if not candidates:
        out.reason = "no configuration reached the frontier"
        return out

    if preference.kind == "constrained":
        kept: list[str] = []
        for config_id in candidates:
            failed = next(
                (
                    bound
                    for bound in preference.subject_to
                    if not bound.satisfied_by(
                        metrics.get(config_id, {}).get(
                            _metric_key(bound.metric)
                        )
                    )
                ),
                None,
            )
            if failed is None:
                kept.append(config_id)
            else:
                out.excluded[config_id] = failed.describe()
        if not kept:
            out.reason = (
                "no configuration on the frontier satisfies "
                + " and ".join(b.describe() for b in preference.subject_to)
            )
            return out
        candidates = kept
        order: Sequence[str] = (preference.optimise,)
    else:
        order = preference.priority

    for objective_id in order:
        survivors = [
            c
            for c in candidates
            if not any(_beats(result, other, c, objective_id) for other in candidates)
        ]
        if survivors:
            candidates = survivors
        if len(candidates) == 1:
            break

    chosen = sorted(candidates)[0]
    out.chosen = chosen
    out.tie_broken = len(candidates) > 1
    out.concedes = _concedes(result, chosen)
    out.reason = (
        f"{chosen} is not beaten on {', '.join(order)} by any config on the frontier"
        + (
            f"; {len(candidates)} configs remained indistinguishable and the "
            "lexicographically first config id was taken"
            if out.tie_broken
            else ""
        )
    )
    return out


def _beats(result: FrontierResult, a: str, b: str, objective_id: str) -> bool:
    """Does ``a`` statistically beat ``b`` on this objective?

    Read from the per-objective comparisons the frontier already computed, so
    a preference cannot disagree with the frontier it is applied to.
    """
    if result.domination is None:
        return False
    verdict = result.domination.verdicts.get((b, a))
    if verdict is None:
        return False
    for comparison in verdict.per_objective:
        if comparison.objective != objective_id:
            continue
        return comparison.p_superior < result.alpha
    return False


def _concedes(result: FrontierResult, chosen: str) -> dict[str, tuple[str, ...]]:
    """Every objective some other frontier config beats the chosen one on."""
    out: dict[str, tuple[str, ...]] = {}
    for objective in result.objectives:
        beaten_by = tuple(
            sorted(
                other
                for other in result.frontier
                if other != chosen and _beats(result, other, chosen, objective.id)
            )
        )
        if beaten_by:
            out[objective.id] = beaten_by
    return out


def objective_ids(objectives: Sequence[Objective]) -> tuple[str, ...]:
    return tuple(o.id for o in objectives)


__all__ += ["Bound", "objective_ids", "resolve"]
