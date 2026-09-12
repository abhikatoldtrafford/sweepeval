"""Operational scorer (spec §11.6).

Rides on calls other scorers already made rather than issuing its own, which is
why it requires no capability and can never be ``SKIPPED``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from sweepeval.capabilities.detect import Capability
from sweepeval.schema.call import Call, ErrorClass
from sweepeval.schema.metric import MetricSpec
from sweepeval.schema.observation import Observation, Verdict
from sweepeval.schema.unit import Unit
from sweepeval.scorers.base import ScoreContext, register

__all__ = ["OperationalScorer", "error_rate", "latency_samples"]


def latency_samples(calls: Sequence[Call]) -> list[float]:
    """Total latency of calls that belong in the population (§6.4).

    Retries and errored calls are excluded. A call that succeeded only after
    30s of backoff describes the rate limiter, and a 500 that took 30s belongs
    in ``error_rate``.
    """
    return [c.timing.total_ms for c in calls if c.counts_toward_latency]


def error_rate(calls: Sequence[Call]) -> float:
    """Terminal and malformed post-retry outcomes over attempted unit-runs.

    Refusals are excluded: they are not errors (§7, §11.6).
    """
    finals = [c for c in calls if c.attempt == 1]
    if not finals:
        return 0.0
    bad = sum(
        1
        for c in finals
        if c.response.error_class in (ErrorClass.terminal, ErrorClass.malformed)
    )
    return bad / len(finals)


@dataclass
class OperationalScorer:
    family: str = "operational"
    version: int = 1
    requires: frozenset[Capability] = field(default_factory=frozenset)

    def metrics(self) -> list[MetricSpec]:
        return [
            MetricSpec(
                metric="latency_p95_ms",
                family="operational",
                direction="minimize",
                unit="ms",
                cluster_key="probe",
            ),
            MetricSpec(
                metric="cost_per_probe",
                family="operational",
                direction="minimize",
                unit="tokens",
                cluster_key="probe",
            ),
            MetricSpec(
                metric="error_rate",
                family="operational",
                direction="minimize",
                unit="rate",
                cluster_key="probe",
            ),
        ]

    def score(
        self, unit: Unit, calls: Sequence[Call], context: ScoreContext
    ) -> list[Observation]:
        samples = latency_samples(calls)
        tokens_out = sum(c.tokens.out or 0 for c in calls)

        def make(
            metric: str, value: float | None, reason: str | None = None
        ) -> Observation:
            return context.observation(
                scorer=self.family,
                version=self.version,
                metric=metric,
                family="operational",
                verdict=Verdict.PASS if value is not None else Verdict.UNSCORABLE,
                value=value,
                reason=reason,
                unit=unit,
                call_ids=[c.unit_id for c in calls],
            )

        latency = (
            make("latency_ms", sum(samples) / len(samples))
            if samples
            else make(
                "latency_ms", None, "no call qualified for the latency population"
            )
        )
        return [
            latency,
            make("tokens_out", float(tokens_out)),
            make("error_rate", error_rate(calls)),
        ]


register(OperationalScorer())
