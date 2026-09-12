"""Scorer protocol and registry (spec §11.1, I10).

I10: adding a scorer touches no runner code. Registration is by decorator
in-tree and by ``entry_points`` out-of-tree, and both go through the same
protocol.

The acceptance test for this interface is a test that registers a fake scorer
and runs it end to end. An interface that has only ever been used by its own
authors is not an interface.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Protocol, runtime_checkable

from sweepeval.capabilities.detect import Capability, CapabilityReport
from sweepeval.schema.call import Call
from sweepeval.schema.metric import MetricSpec
from sweepeval.schema.observation import Observation, Verdict
from sweepeval.schema.unit import Unit

__all__ = [
    "SCORER_ENTRY_POINT_GROUP",
    "ScoreContext",
    "Scorer",
    "ScorerRegistry",
    "register",
    "registry",
]

SCORER_ENTRY_POINT_GROUP = "sweepeval.scorers"


@dataclass
class ScoreContext:
    """Everything a scorer needs that is not the unit or its calls."""

    run_id: str
    config_id: str
    run_idx: int
    text: str
    """Extracted response text of the final turn."""

    texts: Sequence[str] = ()
    """Extracted text of every turn, for multi-turn units."""

    canaries: dict[str, str] = field(default_factory=dict)
    refusal_detected: bool = False
    layer: str = "generic"
    ts: str = ""

    def observation(
        self,
        *,
        scorer: str,
        version: int,
        metric: str,
        family: str,
        verdict: Verdict,
        value: float | None = None,
        reason: str | None = None,
        unit: Unit | None = None,
        call_ids: Sequence[str] = (),
        blob_ids: Sequence[str] = (),
    ) -> Observation:
        """Build an Observation with the run/config/unit keys already filled."""
        return Observation(
            ts=self.ts,
            run_id=self.run_id,
            config_id=self.config_id,
            unit_id=unit.unit_id if unit else "",
            run_idx=self.run_idx,
            scorer=scorer,
            scorer_version=version,
            metric=metric,
            family=family,
            layer=self.layer,  # type: ignore[arg-type]
            verdict=verdict,
            value=value,
            reason=reason,
            severity=unit.severity if unit else None,
            attack_class=unit.attack_class if unit else None,
            policy_id=unit.policy_id if unit else None,
            depth=unit.depth if unit else None,
            call_ids=tuple(call_ids),
            blob_ids=tuple(blob_ids),
        )


@runtime_checkable
class Scorer(Protocol):
    """One scorer family."""

    family: str
    version: int
    requires: frozenset[Capability]

    def metrics(self) -> list[MetricSpec]: ...

    def score(
        self, unit: Unit, calls: Sequence[Call], context: ScoreContext
    ) -> list[Observation]:
        """Return one or more Observations for this unit run."""
        ...


class ScorerRegistry:
    def __init__(self) -> None:
        self._scorers: dict[str, Scorer] = {}

    def register(self, scorer: Scorer) -> Scorer:
        if scorer.family in self._scorers:
            raise ValueError(f"scorer family {scorer.family!r} is already registered")
        self._scorers[scorer.family] = scorer
        return scorer

    def get(self, family: str) -> Scorer:
        return self._scorers[family]

    def all(self) -> tuple[Scorer, ...]:
        return tuple(self._scorers[k] for k in sorted(self._scorers))

    def applicable(self, capabilities: CapabilityReport) -> tuple[Scorer, ...]:
        """Scorers whose required capabilities are all supported."""
        return tuple(
            s
            for s in self.all()
            if all(capabilities.supports(c) for c in s.requires)
        )

    def skipped(
        self, capabilities: CapabilityReport
    ) -> tuple[tuple[Scorer, str], ...]:
        """Scorers that cannot run, each with the reason (I5).

        The reason names the capability detector that ruled it out, so a scorer
        never silently vanishes from a report.
        """
        out: list[tuple[Scorer, str]] = []
        for scorer in self.all():
            missing = [c for c in sorted(scorer.requires, key=lambda c: c.value)
                       if not capabilities.supports(c)]
            if missing:
                out.append(
                    (scorer, "; ".join(capabilities.skip_reason(c) for c in missing))
                )
        return tuple(out)

    def from_entry_points(self) -> None:
        for entry in entry_points(group=SCORER_ENTRY_POINT_GROUP):
            loaded = entry.load()
            scorer = (
                loaded()
                if callable(loaded) and not hasattr(loaded, "family")
                else loaded
            )
            self.register(scorer)


_REGISTRY = ScorerRegistry()


def registry() -> ScorerRegistry:
    return _REGISTRY


def register(scorer: Scorer) -> Scorer:
    return _REGISTRY.register(scorer)
