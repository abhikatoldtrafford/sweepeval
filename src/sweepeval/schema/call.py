"""The ``calls.jsonl`` row (spec §6.2, §6.4).

One row per HTTP call. Operational metrics derive from this file; every other
family derives from ``observations.jsonl``. A depth-15 conversation is fifteen
rows here and one Observation there.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

__all__ = [
    "Call",
    "CallRole",
    "ErrorClass",
    "ExtractionPart",
    "RefusalPart",
    "RequestPart",
    "ResponsePart",
    "TimingPart",
    "TokensPart",
]

CallRole = Literal["target", "judge", "discovery", "capability"]


class ErrorClass(str, Enum):
    """Spec §7's classification table.

    ``refusal`` is deliberately not an error: a target that declines a
    prompt-injection attempt is behaving correctly. ``malformed`` is its own
    class so a body that fails extraction is never scored as a zero.
    """

    ok = "ok"
    retryable = "retryable"
    terminal = "terminal"
    refusal = "refusal"
    malformed = "malformed"


class RequestPart(BaseModel):
    model_config = ConfigDict(frozen=True)

    shape: str
    params_hash: str
    body_sha256: str
    bytes: int


class ResponsePart(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: int
    error_class: ErrorClass
    streamed: bool
    bytes: int
    body_sha256: str | None = None


class TimingPart(BaseModel):
    """Timings in milliseconds.

    ``total_ms`` **excludes** ``queue_ms``. Under concurrency the harness's own
    queueing would otherwise be measured as target latency, which would make
    ``latency_p95_ms`` partly a measurement of sweepeval.
    """

    model_config = ConfigDict(frozen=True)

    queue_ms: float
    ttft_ms: float | None
    total_ms: float


class TokensPart(BaseModel):
    """Token counts and their provenance (§12.4).

    ``reasoning`` is separate from ``out``: reasoning tokens are billed and, on
    most providers, reported apart from the answer (§7).
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    in_: int | None = Field(default=None, alias="in")
    out: int | None = None
    reasoning: int | None = None
    source: Literal["MEASURED", "ESTIMATED"] = "ESTIMATED"


class ExtractionPart(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str | None
    ok: bool
    text_sha256: str | None = None
    text_len: int | None = None


class RefusalPart(BaseModel):
    model_config = ConfigDict(frozen=True)

    detected: bool = False
    score: float | None = None


class Call(BaseModel):
    """One HTTP call."""

    model_config = ConfigDict(frozen=True)

    ts: str
    run_id: str
    config_id: str
    unit_id: str
    run_idx: int
    turn_idx: int
    attempt: int
    role: CallRole = "target"
    request: RequestPart
    response: ResponsePart
    timing: TimingPart
    tokens: TokensPart
    extraction: ExtractionPart
    refusal: RefusalPart

    @computed_field  # type: ignore[prop-decorator]
    @property
    def counts_toward_latency(self) -> bool:
        """Whether this call belongs in the latency population (§6.4, §13.3).

        Retries are excluded — a call that succeeded only after 30s of backoff
        describes the rate limiter, not the target. Errored calls are excluded
        for the same reason: a 500 that took 30s is a failure, and it belongs
        in ``error_rate``, not in ``latency_p95_ms``.
        """
        return self.attempt == 1 and self.response.error_class is ErrorClass.ok

    @classmethod
    def example(cls, **over: Any) -> Call:
        """Fixture helper for tests. Not part of the public surface."""
        base: dict[str, Any] = dict(
            ts="2026-09-12T00:00:00Z",
            run_id="r1",
            config_id="c1",
            unit_id="u#0123456789abcdef",
            run_idx=0,
            turn_idx=0,
            attempt=1,
            request=RequestPart(
                shape="openai.chat_completions",
                params_hash="0" * 16,
                body_sha256="0" * 64,
                bytes=10,
            ),
            response=ResponsePart(
                status=200,
                error_class=ErrorClass.ok,
                streamed=False,
                bytes=20,
                body_sha256="1" * 64,
            ),
            timing=TimingPart(queue_ms=0.0, ttft_ms=10.0, total_ms=100.0),
            tokens=TokensPart(),
            refusal=RefusalPart(),
            extraction=ExtractionPart(
                path="$.x", ok=True, text_sha256="2" * 64, text_len=5
            ),
        )
        base.update(over)
        return cls(**base)
