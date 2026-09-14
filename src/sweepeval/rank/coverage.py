"""Coverage parity (spec §14.5). I4-adjacent.

A config whose context window failed every depth-15 conversation has roughly
29% missing coverage on that family. Comparing it against a config with full
coverage compares different things: the paired test pairs on the probe, and a
probe only one side scored is not a matched block. The difference it produces
is a statement about which probes survived, not about the configs.

Two thresholds:

* **>10% divergence blocks domination for that pair.** Not for the config —
  for the pair. A config with a gap can still be compared against another
  config with the same gap, because there the matched blocks line up again.
* **>30% flags LOW_COVERAGE and excludes the metric** from the objective, for
  every pair.

The comparison is on the *scored* count per family, not the attempted count.
An attempted-but-unscorable trial contributes nothing to the estimate, so
counting it as coverage would hide exactly the gap this exists to find.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from sweepeval.schema.observation import Observation, Verdict

__all__ = [
    "BLOCK_THRESHOLD",
    "EXCLUDE_THRESHOLD",
    "CoverageMatrix",
    "FamilyCoverage",
    "count_coverage",
    "coverage_from_counts",
    "coverage_matrix",
]

BLOCK_THRESHOLD = 0.10
"""Divergence above this blocks domination between the pair (§14.5)."""

EXCLUDE_THRESHOLD = 0.30
"""Divergence above this flags LOW_COVERAGE and drops the metric."""


@dataclass(frozen=True)
class FamilyCoverage:
    config_id: str
    family: str
    scored: int
    attempted: int

    @property
    def rate(self) -> float:
        return self.scored / self.attempted if self.attempted else 0.0


@dataclass
class CoverageMatrix:
    """Per-config, per-family scored counts and the divergences between them."""

    families: tuple[str, ...] = ()
    rows: dict[tuple[str, str], FamilyCoverage] = field(default_factory=dict)

    def rate(self, config_id: str, family: str) -> float:
        row = self.rows.get((config_id, family))
        return row.rate if row else 0.0

    def divergence(self, a: str, b: str, family: str) -> float:
        """Absolute difference in scored rate between two configs."""
        return abs(self.rate(a, family) - self.rate(b, family))

    def blocks(self, a: str, b: str, family: str) -> bool:
        return self.divergence(a, b, family) > BLOCK_THRESHOLD

    def excludes(self, a: str, b: str, family: str) -> bool:
        return self.divergence(a, b, family) > EXCLUDE_THRESHOLD

    def blocked_families(self, a: str, b: str) -> tuple[str, ...]:
        return tuple(f for f in self.families if self.blocks(a, b, f))

    def excluded_families(self, a: str, b: str) -> tuple[str, ...]:
        return tuple(f for f in self.families if self.excludes(a, b, f))

    def worst(self) -> tuple[str, float]:
        """The family with the widest spread across configs, for the report."""
        worst_family, worst_gap = "", 0.0
        for family in self.families:
            rates = [
                r.rate for (_, f), r in self.rows.items() if f == family and r.attempted
            ]
            if len(rates) < 2:
                continue
            gap = max(rates) - min(rates)
            if gap > worst_gap:
                worst_family, worst_gap = family, gap
        return worst_family, worst_gap


def coverage_from_counts(
    counts: Mapping[str, Mapping[str, tuple[int, int]]],
    families: Sequence[str] = ("security", "guardrail", "determinism", "context"),
) -> CoverageMatrix:
    """Build the matrix from stored ``(scored, attempted)`` pairs.

    A stored run has aggregates, not observations. Re-reading
    ``observations.jsonl`` to recount would make offline reporting depend on
    the full log, so the counts are summarised into ``aggregates.json`` and
    read back here — the same numbers, without the log.
    """
    matrix = CoverageMatrix(families=tuple(families))
    for config_id, per_family in counts.items():
        for family in families:
            scored, attempted = per_family.get(family, (0, 0))
            matrix.rows[(config_id, family)] = FamilyCoverage(
                config_id=config_id,
                family=family,
                scored=int(scored),
                attempted=int(attempted),
            )
    return matrix


def count_coverage(
    observations: Iterable[Observation],
    families: Sequence[str] = ("security", "guardrail", "determinism", "context"),
) -> dict[str, tuple[int, int]]:
    """``family -> (scored, attempted)`` for one config."""
    counts: dict[str, list[int]] = {f: [0, 0] for f in families}
    # One count per trial, however many times it was scored. §11.9's judge
    # appends a second observation for an ambiguity rather than rewriting the
    # first (I7), so counting rows reported 14 scored of 34 attempted on a
    # family with 20 trials -- inflating the denominator precisely where the
    # judge had just improved the numerator.
    for observation in _latest_per_trial(observations):
        if observation.family not in counts:
            continue
        counts[observation.family][1] += 1
        if observation.verdict not in (Verdict.UNSCORABLE, Verdict.SKIPPED):
            counts[observation.family][0] += 1
    return {f: (v[0], v[1]) for f, v in counts.items()}


def _latest_per_trial(observations: Iterable[Observation]) -> list[Observation]:
    """Collapse re-scorings of the same ``(metric, unit, run)``.

    Precedence by verdict rather than by scorer name: a decided verdict
    supersedes an UNSCORABLE, and among decided rows the later wins. `rank`
    does not need to know the judge exists, and the rule holds for any future
    re-scorer.
    """
    winners: dict[tuple[str, str, int], Observation] = {}
    order: list[tuple[str, str, int]] = []

    def decided(o: Observation) -> bool:
        return o.verdict not in (Verdict.UNSCORABLE, Verdict.SKIPPED)

    for observation in observations:
        key = (observation.metric, observation.unit_id, observation.run_idx)
        current = winners.get(key)
        if current is None:
            winners[key] = observation
            order.append(key)
        elif decided(observation) or not decided(current):
            winners[key] = observation

    return [winners[k] for k in order]


def coverage_matrix(
    per_config: Mapping[str, Iterable[Observation]],
    families: Sequence[str] = ("security", "guardrail", "determinism", "context"),
) -> CoverageMatrix:
    """Count scored and attempted trials per (config, family)."""
    matrix = CoverageMatrix(families=tuple(families))

    for config_id, observations in per_config.items():
        counts: dict[str, list[int]] = {f: [0, 0] for f in families}
        for observation in observations:
            family = observation.family
            if family not in counts:
                continue
            counts[family][1] += 1
            if observation.verdict not in (Verdict.UNSCORABLE, Verdict.SKIPPED):
                counts[family][0] += 1

        for family, (scored, attempted) in counts.items():
            matrix.rows[(config_id, family)] = FamilyCoverage(
                config_id=config_id,
                family=family,
                scored=scored,
                attempted=attempted,
            )

    return matrix
