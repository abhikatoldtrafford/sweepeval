"""The ``calls.jsonl`` row (spec §6.2, §6.4).

One row per HTTP call. Operational metrics derive from this file; every other
family derives from ``observations.jsonl``. A depth-15 conversation is fifteen
rows here and one Observation there.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from sweepeval.schema.hashing import sha256_hex

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

    error_excerpt: str | None = None
    """The first part of a failed response's body, redacted like everything else.

    Only set when the call failed. A status code cannot tell "your prompt is
    longer than my context window" from "your JSON is malformed", and the
    degradation scorer has to tell those apart: the first is a finding about
    the target and the second is a bug in this tool. Everything else about the
    body is a hash, which is unreadable precisely when a reader needs it most.

    Truncated, and it passes through the store's redactor with the rest of the
    row -- an error body is a common place for a provider to echo a request
    header back.
    """


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

    in_flight: int = 1
    """How many of this run's requests were in flight when this one was sent.

    ``1`` for everything the executor issues, which is everything except the
    degradation family's concurrency ramp (§11, family 8). The ramp is the
    only place the tool deliberately contends with itself, and a call sent
    against seven others is measuring sweepeval's own queueing as much as the
    target -- which is exactly why it is excluded from the latency population
    below, and why the number is recorded rather than inferred.
    """

    @computed_field  # type: ignore[prop-decorator]
    @property
    def counts_toward_latency(self) -> bool:
        """Whether this call belongs in the latency population (§6.4, §13.3).

        Retries are excluded — a call that succeeded only after 30s of backoff
        describes the rate limiter, not the target. Errored calls are excluded
        for the same reason: a 500 that took 30s is a failure, and it belongs
        in ``error_rate``, not in ``latency_p95_ms``.

        So are ramped calls. `latency_p95_ms` is a measurement of the target,
        and the governor's own docstring says why: two in-flight requests make
        part of it a measurement of our queueing. The degradation ramp sends
        many on purpose, and the operational scorer scores every unit's calls
        — including this family's — so without this the ramp would silently
        move the latency number of every run that included it.
        """
        return (
            self.attempt == 1
            and self.response.error_class is ErrorClass.ok
            and self.in_flight <= 1
        )

    def with_extraction(self, path: str | None, text: str) -> Call:
        """Record what extraction actually found (§6.2).

        The transport writes ``ok=False`` on every row because it cannot know
        better: the declared path is chosen a layer above it, and for a
        non-streamed response the transport never reads the body at all.
        Extraction happens in the runner and, until this method existed,
        nothing wrote the answer back -- so every stored row claimed the text
        could not be read while the scorers were scoring that very text.

        That is worse than an unused field. §5.1 promises a stored run can be
        re-scored offline from these rows, and anyone filtering on
        ``extraction.ok`` to find the responses that failed to parse would
        have selected the whole run.
        """
        return self.model_copy(
            update={
                "extraction": ExtractionPart(
                    path=path,
                    ok=bool(text),
                    text_sha256=sha256_hex(text.encode()) if text else None,
                    text_len=len(text),
                )
            }
        )

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
