"""Objective correlation (spec §14.3).

Six objectives probably span fewer than six dimensions. Latency and cost both
track output length; security and guardrail move together; retention is
largely a property of the target that barely responds to the swept axes. An
objective that does not respond to the axes contributes no discrimination
while adding a dimension in which nothing can be dominated — which is one of
the ways a frontier ends up containing everything.

The report prints the empirical matrix rather than hiding the problem, and
names the pairs above the threshold so the user can drop one with
``--objectives``.

Lives in ``stats`` because the layering contract keeps numerics here: nothing
outside this package computes an interval, a p-value, or — as of this module —
a correlation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["HIGH_CORRELATION", "CorrelationMatrix", "correlation_matrix"]

HIGH_CORRELATION = 0.9
"""§14.3's threshold for saying two objectives are measuring one thing."""

MIN_CONFIGS = 3
"""Below three points a correlation is not a summary of anything: two points
are perfectly correlated by construction, whatever they are."""


@dataclass
class CorrelationMatrix:
    objectives: tuple[str, ...] = ()
    values: dict[tuple[str, str], float] | None = None
    reason: str = ""

    def get(self, a: str, b: str) -> float | None:
        if not self.values:
            return None
        return self.values.get((a, b)) or self.values.get((b, a))

    def high_pairs(
        self, threshold: float = HIGH_CORRELATION
    ) -> tuple[tuple[str, str, float], ...]:
        if not self.values:
            return ()
        out = [
            (a, b, v)
            for (a, b), v in sorted(self.values.items())
            if a < b and abs(v) >= threshold
        ]
        return tuple(sorted(out, key=lambda t: -abs(t[2])))

    @property
    def computed(self) -> bool:
        return bool(self.values)


def correlation_matrix(
    points: Mapping[str, Mapping[str, float]],
    objectives: Sequence[str],
) -> CorrelationMatrix:
    """Pearson correlation of each objective pair across the swept configs.

    ``points`` is ``config_id -> objective -> value``. Objectives missing from
    any config, or constant across all of them, are excluded and the reason is
    reported: a constant objective has zero variance, and dividing by it
    produces a NaN that would print as a number.
    """
    config_ids = sorted(points)
    if len(config_ids) < MIN_CONFIGS:
        return CorrelationMatrix(
            reason=(
                f"{len(config_ids)} config(s): a correlation over fewer than "
                f"{MIN_CONFIGS} points summarises nothing"
            )
        )

    usable: list[str] = []
    columns: list[list[float]] = []
    for objective in objectives:
        column = [points[c].get(objective) for c in config_ids]
        if any(v is None for v in column):
            continue
        floats = [float(v) for v in column if v is not None]
        if max(floats) - min(floats) == 0.0:
            # Constant across the sweep. Not an error — §14.2 predicts exactly
            # this for target_determinism_at_temp0 within a temperature triple
            # — but it has no correlation with anything.
            continue
        usable.append(objective)
        columns.append(floats)

    if len(usable) < 2:
        return CorrelationMatrix(
            objectives=tuple(usable),
            reason=(
                "fewer than two objectives vary across the sweep, so there is "
                "no correlation to report"
            ),
        )

    matrix = np.corrcoef(np.array(columns, dtype=float))
    values: dict[tuple[str, str], float] = {}
    for i, a in enumerate(usable):
        for j, b in enumerate(usable):
            value = float(matrix[i][j])
            values[(a, b)] = 0.0 if np.isnan(value) else round(value, 4)

    return CorrelationMatrix(objectives=tuple(usable), values=values)
