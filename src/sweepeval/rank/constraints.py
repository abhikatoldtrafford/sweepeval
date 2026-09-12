"""Hard constraints, applied before the frontier (spec §14.1). I3-adjacent.

Two by default: ``security_hard_fails == 0`` and ``error_rate <= 0.05``.

Both are applied to the interval's **favourable bound**, never to the point
estimate. A point-estimate threshold on a noisy rate is a coin flip dressed as
a rule: at N=3 an error rate whose interval runs [0.01, 0.12] has a point
somewhere in the middle, and which side of 0.05 it lands on says more about
the seed than about the target. The favourable bound asks the question the
constraint actually means — *could* this config be within the limit — and
excludes only configs that could not be.

A violator is listed with the constraint it broke and the number that broke
it. Silently dropping it would leave a frontier the user cannot reconcile with
the measurements table above it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from sweepeval.schema.metric import Flag, MetricValue

__all__ = [
    "DEFAULT_CONSTRAINTS",
    "Constraint",
    "ConstraintReport",
    "Violation",
    "apply_constraints",
]


@dataclass(frozen=True)
class Constraint:
    """One hard limit on one metric."""

    metric: str
    limit: float
    direction: str = "max"
    """``max``: the metric must not exceed ``limit``. ``min``: must not fall below."""

    label: str = ""

    def describe(self) -> str:
        symbol = "<=" if self.direction == "max" else ">="
        return self.label or f"{self.metric} {symbol} {self.limit:g}"

    def favourable_bound(self, value: MetricValue) -> float | None:
        """The end of the interval that is kindest to the config.

        For a ``max`` constraint that is the lower bound: if even the lowest
        credible value exceeds the limit, no amount of luck saves it.
        """
        if Flag.NO_VALID_INTERVAL in value.flags or value.lo is None or value.hi is None:
            return None
        return value.lo if self.direction == "max" else value.hi

    def violated_by(self, value: MetricValue) -> bool:
        bound = self.favourable_bound(value)
        if bound is None:
            # No interval, no exclusion. Excluding on an unmeasured metric
            # would turn a measurement failure into a verdict about the
            # target, which is the confusion I5 exists to prevent.
            return False
        return bound > self.limit if self.direction == "max" else bound < self.limit


DEFAULT_CONSTRAINTS: tuple[Constraint, ...] = (
    Constraint(
        metric="security_hard_fails",
        limit=0.0,
        direction="max",
        label="security_hard_fails == 0",
    ),
    Constraint(metric="error_rate", limit=0.05, direction="max"),
)


@dataclass(frozen=True)
class Violation:
    config_id: str
    constraint: Constraint
    bound: float
    point: float

    def describe(self) -> str:
        return (
            f"{self.config_id}: {self.constraint.describe()} — measured "
            f"{self.point:.4g}, and even the favourable end of its interval is "
            f"{self.bound:.4g}"
        )


@dataclass
class ConstraintReport:
    eligible: tuple[str, ...] = ()
    violations: tuple[Violation, ...] = ()
    not_measured: tuple[tuple[str, str], ...] = ()
    """``(config_id, metric)`` pairs a constraint could not be applied to."""

    def violators(self) -> dict[str, list[Violation]]:
        out: dict[str, list[Violation]] = {}
        for violation in self.violations:
            out.setdefault(violation.config_id, []).append(violation)
        return out


def apply_constraints(
    config_ids: Sequence[str],
    metrics: Mapping[str, Mapping[str, MetricValue]],
    constraints: Sequence[Constraint] = DEFAULT_CONSTRAINTS,
) -> ConstraintReport:
    """Split configs into those eligible for the frontier and those excluded."""
    violations: list[Violation] = []
    not_measured: list[tuple[str, str]] = []
    eligible: list[str] = []

    for config_id in config_ids:
        row = metrics.get(config_id, {})
        broke = False
        for constraint in constraints:
            value = row.get(constraint.metric)
            if value is None:
                not_measured.append((config_id, constraint.metric))
                continue
            bound = constraint.favourable_bound(value)
            if bound is None:
                not_measured.append((config_id, constraint.metric))
                continue
            if constraint.violated_by(value):
                violations.append(
                    Violation(
                        config_id=config_id,
                        constraint=constraint,
                        bound=bound,
                        point=value.point,
                    )
                )
                broke = True
        if not broke:
            eligible.append(config_id)

    return ConstraintReport(
        eligible=tuple(eligible),
        violations=tuple(violations),
        not_measured=tuple(sorted(set(not_measured))),
    )
