"""The deferred-family mechanism (spec §11.10, D2).

A family the spec declares and this build does not implement registers a
**real** scorer object emitting ``SKIPPED: not_implemented``, so it appears in
every report. A family absent from a report is indistinguishable from one that
passed.

Nothing uses it right now: tool integrity, retrieval and degradation were the
three, and all three are built. Two of them were deferred partly on capability
detectors that reported UNSUPPORTED against endpoints which supported the
capability perfectly well -- the tool probe never offered a tool, and the
citation probe looked for keys a real retrieval endpoint does not use. Being
loudly absent is what made those readings checkable at all.

The reason string names no version and the briefs promise no release. They
used to say ``not_implemented_in_v0.1`` and "lands in v0.2"; 0.2 then shipped
without them, so the tool announced a broken promise on every run, in a
machine-readable string a user could reasonably have planned around.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from sweepeval.capabilities.detect import Capability
from sweepeval.schema.call import Call
from sweepeval.schema.metric import MetricSpec
from sweepeval.schema.observation import Observation, Verdict
from sweepeval.schema.unit import Unit
from sweepeval.scorers.base import ScoreContext

__all__ = ["DEFERRED_REASON", "DeferredScorer"]

DEFERRED_REASON = "not_implemented"
"""Machine-readable, and deliberately version-free.

Observations stored by earlier runs keep whatever string they were written
with; this governs new ones only, so an offline re-report of an old run still
shows what that run actually said.
"""


@dataclass
class DeferredScorer:
    """A family the spec declares and this build does not implement."""

    family: str = ""
    version: int = 0
    requires: frozenset[Capability] = field(default_factory=frozenset)
    metric_name: str = ""
    brief: str = ""

    def metrics(self) -> list[MetricSpec]:
        return [
            MetricSpec(
                metric=self.metric_name,
                family=self.family,
                direction="maximize",
                unit="rate",
                cluster_key="probe",
            )
        ]

    def score(
        self, unit: Unit, calls: Sequence[Call], context: ScoreContext
    ) -> list[Observation]:
        return [
            context.observation(
                scorer=self.family,
                version=self.version,
                metric=self.metric_name,
                family=self.family,
                verdict=Verdict.SKIPPED,
                reason=f"{DEFERRED_REASON}: {self.brief}",
                unit=unit,
            )
        ]


# Nothing is registered here any more. All three families the spec deferred --
# tool integrity, retrieval and degradation -- are built.
#
# `DeferredScorer` stays, and not out of sentiment. It is the mechanism a
# family uses to be *visibly* absent instead of silently missing, and the next
# one to be specified ahead of being built will need it. Two of the three were
# deferred on capability readings that turned out to be wrong, which is an
# argument for keeping the announcement cheap rather than for deleting it.
