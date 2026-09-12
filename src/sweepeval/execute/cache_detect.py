"""Cache detection (spec §12.7, D41).

N identical requests against an endpoint with prompt caching or a semantic
cache proxy return the cached response for runs 2 and 3. That falsifies
determinism (it goes to 1.0) and deflates latency — two default objectives —
and nothing in the response says so.

**Identical responses are not the signal.** A deterministic target at
temperature 0 returns identical responses too, and that is the thing
``target_determinism_at_temp0`` exists to measure; flagging it would make the
flag fire on exactly the result it is supposed to qualify. The signal is
identical responses *whose repeats arrive too fast to have been generated* —
a request body that was hashed before, a byte-identical response, and a
latency that collapses against the first run's.

Per-run canaries (§11.2) already differ, so security units send different
bodies on every run and cannot exhibit the pattern at all. This catches the
rest: determinism, context and guardrail units, whose bodies are constant by
design.

The flag is a qualifier, not a verdict. It says the determinism and latency
numbers from this config are not trustworthy, and the report says so beside
them rather than silently discarding them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from sweepeval.schema.call import Call, ErrorClass
from sweepeval.schema.metric import Flag, MetricValue

__all__ = [
    "AFFECTED_METRICS",
    "LATENCY_COLLAPSE_RATIO",
    "MIN_DROP_MS",
    "MIN_EVIDENCE_GROUPS",
    "CacheEvidence",
    "CacheVerdict",
    "detect_cache",
    "flag_affected",
]

LATENCY_COLLAPSE_RATIO = 0.25
"""A repeat is implausible when it arrives in under a quarter of the first
run's time. Generation cost does not vary fourfold between two identical
requests to the same model; a cache hit does exactly that."""

MIN_DROP_MS = 20.0
"""The collapse must also be a real amount of time.

An absolute "faster than any model could answer" floor was tried and rejected:
it fires on every localhost and self-hosted target, where a 5ms answer is
normal and means nothing. The ratio is scale-free, and this floor only keeps
scheduling noise on an already-fast endpoint from clearing it — a drop from
8ms to 1ms is a quarter of the time and none of the evidence."""

MIN_EVIDENCE_GROUPS = 2
"""One fast repeat is noise — a warm connection, a scheduling gap, a lucky
queue. Two independent request bodies showing the same collapse is a pattern.
Requiring two is what keeps the flag from firing on every run and thereby
meaning nothing."""

AFFECTED_METRICS: frozenset[str] = frozenset(
    {
        "target_determinism_at_temp0",
        "config_repeatability",
        "semantic_stability",
        "invariance",
        "latency_ms",
        "latency_p95_ms",
        "ttft_ms",
    }
)
"""The metrics a cache corrupts. Pass rates are not here: a cached refusal is
still a refusal, and the security and guardrail rates remain true statements
about what the target returns."""


@dataclass(frozen=True)
class CacheEvidence:
    """One request body that was answered identically, and too fast, twice."""

    config_id: str
    unit_id: str
    turn_idx: int
    body_sha256: str
    response_sha256: str
    first_run: int
    repeat_run: int
    first_ms: float
    repeat_ms: float
    measure: str

    def describe(self) -> str:
        return (
            f"{self.unit_id} turn {self.turn_idx}: run {self.repeat_run} returned "
            f"the same body in {self.repeat_ms:.0f}ms against run "
            f"{self.first_run}'s {self.first_ms:.0f}ms ({self.measure})"
        )


@dataclass
class CacheVerdict:
    config_id: str = ""
    suspected: bool = False
    evidence: tuple[CacheEvidence, ...] = ()
    reason: str = "no repeated request body was answered implausibly fast"
    groups_examined: int = 0

    @property
    def flags(self) -> tuple[Flag, ...]:
        return (Flag.CACHE_SUSPECTED,) if self.suspected else ()


@dataclass
class _Group:
    """Every attempt at one ``(unit, turn, request body)``, keyed by run."""

    body_sha256: str
    by_run: dict[int, Call] = field(default_factory=dict)


def detect_cache(calls: Iterable[Call], *, config_id: str = "") -> CacheVerdict:
    """Look for identical bodies answered identically and implausibly fast."""
    groups: dict[tuple[str, int, str], _Group] = {}

    for call in calls:
        if config_id and call.config_id != config_id:
            continue
        # Only first attempts: a retry's timing describes the backoff, and a
        # retry after an error has no first-run pair to compare against.
        if call.attempt != 1 or call.response.error_class is not ErrorClass.ok:
            continue
        key = (call.unit_id, call.turn_idx, call.request.body_sha256)
        group = groups.setdefault(key, _Group(call.request.body_sha256))
        # First attempt of a run wins; a run should not appear twice, but if
        # it does the earliest is the one whose timing is comparable.
        group.by_run.setdefault(call.run_idx, call)

    evidence: list[CacheEvidence] = []
    repeated = 0

    for (unit_id, turn_idx, body_sha), group in sorted(groups.items()):
        runs = sorted(group.by_run)
        if len(runs) < 2:
            continue
        repeated += 1
        first = group.by_run[runs[0]]
        for run_idx in runs[1:]:
            repeat = group.by_run[run_idx]
            if not _same_response(first, repeat):
                continue
            first_ms, repeat_ms, measure = _timings(first, repeat)
            if first_ms is None or repeat_ms is None:
                continue
            if not _implausible(first_ms, repeat_ms):
                continue
            evidence.append(
                CacheEvidence(
                    config_id=config_id or first.config_id,
                    unit_id=unit_id,
                    turn_idx=turn_idx,
                    body_sha256=body_sha,
                    response_sha256=repeat.response.body_sha256 or "",
                    first_run=runs[0],
                    repeat_run=run_idx,
                    first_ms=first_ms,
                    repeat_ms=repeat_ms,
                    measure=measure,
                )
            )

    bodies = {e.body_sha256 for e in evidence}
    suspected = len(bodies) >= MIN_EVIDENCE_GROUPS

    if suspected:
        reason = (
            f"{len(bodies)} repeated request bodies were answered byte-identically "
            f"and implausibly fast; determinism and latency from this config are "
            f"not trustworthy (§12.7)"
        )
    elif bodies:
        reason = (
            f"{len(bodies)} repeated body answered implausibly fast — below the "
            f"{MIN_EVIDENCE_GROUPS}-body threshold, so this is treated as noise"
        )
    elif repeated:
        reason = (
            f"{repeated} request bodies repeated across runs, none answered "
            f"implausibly fast"
        )
    else:
        reason = "no request body was sent more than once (per-run canaries differ)"

    return CacheVerdict(
        config_id=config_id,
        suspected=suspected,
        evidence=tuple(evidence),
        reason=reason,
        groups_examined=repeated,
    )


def _same_response(first: Call, repeat: Call) -> bool:
    """Byte-identical response, positively established.

    Two calls with no recorded response hash are not evidence of anything, so
    a missing hash is a non-match rather than a match by default.
    """
    a = first.response.body_sha256
    b = repeat.response.body_sha256
    return bool(a) and a == b


def _timings(first: Call, repeat: Call) -> tuple[float | None, float | None, str]:
    """TTFT when both calls have it, otherwise total. Never mixed.

    Comparing one call's TTFT against another's total would manufacture a
    collapse out of a streaming difference.
    """
    if first.timing.ttft_ms is not None and repeat.timing.ttft_ms is not None:
        return first.timing.ttft_ms, repeat.timing.ttft_ms, "ttft"
    return first.timing.total_ms, repeat.timing.total_ms, "total"


def _implausible(first_ms: float, repeat_ms: float) -> bool:
    if first_ms <= 0:
        return False
    if first_ms - repeat_ms < MIN_DROP_MS:
        return False
    return repeat_ms <= first_ms * LATENCY_COLLAPSE_RATIO


def flag_affected(
    metrics: Mapping[str, MetricValue], verdict: CacheVerdict
) -> dict[str, MetricValue]:
    """Attach ``CACHE_SUSPECTED`` to every metric a cache would corrupt.

    Returns a new mapping. ``MetricValue`` is frozen, and rewriting one in
    place would let a flag appear on a value already serialised without it.
    """
    if not verdict.suspected:
        return dict(metrics)

    out: dict[str, MetricValue] = {}
    for name, value in metrics.items():
        if name in AFFECTED_METRICS and Flag.CACHE_SUSPECTED not in value.flags:
            out[name] = value.model_copy(
                update={"flags": (*value.flags, Flag.CACHE_SUSPECTED)}
            )
        else:
            out[name] = value
    return out


def describe(verdict: CacheVerdict, *, limit: int = 3) -> Sequence[str]:
    """Lines for the report: the reason, then the first few examples."""
    lines = [verdict.reason]
    lines.extend(e.describe() for e in verdict.evidence[:limit])
    if len(verdict.evidence) > limit:
        lines.append(f"... and {len(verdict.evidence) - limit} more")
    return lines


__all__ += ["describe"]
