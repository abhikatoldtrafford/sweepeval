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


#: Outcomes that count against the target once retries are done with.
FAILED_CLASSES = (ErrorClass.terminal, ErrorClass.malformed, ErrorClass.retryable)
"""``retryable`` is here because this is the **post-retry** outcome.

A call still classed retryable after the governor gave up did not succeed. It
was excluded before, and combined with counting only first attempts that made
the metric structurally incapable of reporting a failure: against an endpoint
returning 429 to all 160 calls, ``error_rate`` was 0.0 with no LOW_COVERAGE
flag -- a totally unavailable target indistinguishable from a flawless one,
on a metric that is also a default hard constraint.
"""


def final_attempts(calls: Sequence[Call]) -> list[Call]:
    """The last attempt at each ``(unit, run, turn)``.

    First attempts are the wrong population: a call that failed twice and
    succeeded on the third try is a success, and one that failed three times
    is a failure. Only the last attempt says which.
    """
    latest: dict[tuple[str, int, int], Call] = {}
    for call in calls:
        key = (call.unit_id, call.run_idx, call.turn_idx)
        seen = latest.get(key)
        if seen is None or call.attempt > seen.attempt:
            latest[key] = call
    return [latest[k] for k in sorted(latest)]


def error_rate(calls: Sequence[Call]) -> float | None:
    """Failed post-retry outcomes over the turns that were attempted.

    Refusals are excluded: they are not errors (§7, §11.6). ``None`` means no
    turn was attempted at all, which is unscorable rather than perfect.
    """
    finals = final_attempts(calls)
    if not finals:
        return None
    bad = sum(1 for c in finals if c.response.error_class in FAILED_CLASSES)
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
        # The same calls `cost.account` bills for: first attempt, not
        # terminal. Summing every call counted retries and dead requests into
        # the ranked cost axis while the printed cost block excluded them, so
        # the two disagreed about the same configuration.
        billable = [
            c
            for c in calls
            if c.attempt == 1 and c.response.error_class is not ErrorClass.terminal
        ]
        tokens_out = sum(c.tokens.out or 0 for c in billable)
        tokens_in = sum(c.tokens.in_ or 0 for c in billable)
        reasoning = sum(c.tokens.reasoning or 0 for c in billable)

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
        rate = error_rate(calls)
        return [
            latency,
            make("tokens_out", float(tokens_out)),
            # Input and reasoning tokens are billed too. Without them the cost
            # objective could only ever rank output tokens, so a reasoning
            # model that answers tersely looked cheaper than a plain model
            # that answers at length -- the opposite of the invoice.
            make("tokens_in", float(tokens_in)),
            make("tokens_reasoning", float(reasoning)),
            (
                make("error_rate", rate)
                if rate is not None
                else make("error_rate", None, "no turn was attempted")
            ),
        ]


register(OperationalScorer())
