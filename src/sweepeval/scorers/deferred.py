"""Deferred scorer families (spec §11.10, D2).

Tool integrity, retrieval and degradation register **real** scorer objects that
emit ``SKIPPED: not_implemented_in_v0.1``.

Registering them rather than omitting them does two things. It proves the
plugin interface against the hardest cases — schema-persisting, N-run,
concurrency-ramping scorers — before an external contributor meets it. And it
keeps the output honest: a family absent from a report is indistinguishable
from a family that passed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from sweepeval.capabilities.detect import Capability
from sweepeval.schema.call import Call
from sweepeval.schema.metric import MetricSpec
from sweepeval.schema.observation import Observation, Verdict
from sweepeval.schema.unit import Unit
from sweepeval.scorers.base import ScoreContext, register

__all__ = ["DEFERRED_REASON", "DeferredScorer"]

DEFERRED_REASON = "not_implemented_in_v0.1"


@dataclass
class DeferredScorer:
    """A family declared but not implemented in this version."""

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


for _scorer in (
    DeferredScorer(
        family="tool_integrity",
        metric_name="tool_schema_stability",
        requires=frozenset({Capability.TOOL_CALLING}),
        brief=(
            "schema drift, malformed-call rate and tool-selection stability "
            "land in v0.2 (spec section 11, family 4)"
        ),
    ),
    DeferredScorer(
        family="retrieval",
        metric_name="retrieval_ndcg",
        requires=frozenset({Capability.RETRIEVAL}),
        brief=(
            "precision@k, recall@k, MRR and nDCG land in v0.2 "
            "(spec section 11, family 6)"
        ),
    ),
    DeferredScorer(
        family="degradation",
        metric_name="degradation_resilience",
        brief=(
            "concurrency ramp, long inputs and induced tool failures land in "
            "v0.2 (spec section 11, family 8)"
        ),
    ),
):
    register(_scorer)

del _scorer
