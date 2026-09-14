"""Rate limiting, retry, circuit breaking and budget (spec §7, D25).

The governor owns every decision about *when* a request may be sent and
*whether* a failure is worth retrying. The runner simply awaits it. That
separation is deliberate: rate-limiting logic scattered through the runner is
how a tool ends up hammering a production endpoint by accident, and this tool
points an attack suite at endpoints people depend on.

Conservative by default: concurrency 2, four attempts, full jitter, and a
circuit breaker that gives up on a config rather than burning the whole sweep.

**Judgement call, flagged by the plan:** jitter distribution is unspecified.
This uses full jitter — ``uniform(0, base * 2**attempt)`` — capped, and seeded
from the master seed so backoff is reproducible in tests.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field

from sweepeval.schema.call import ErrorClass

__all__ = [
    "BASE_DELAY_S",
    "CIRCUIT_BREAKER_THRESHOLD",
    "DEFAULT_CONCURRENCY",
    "MAX_ATTEMPTS",
    "MAX_DELAY_S",
    "BudgetExceeded",
    "Governor",
]

DEFAULT_CONCURRENCY = 1
"""One request in flight at a time, which is what the executor has always done.

This said 2 for a long while and nothing dispatched two. The semaphore was
built, sized and never contended: measured against a transport that leaves the
event loop free, peak in-flight over a 60-call run is 1, and ``queue_ms`` is 0
on every row of a 4,000-call live sweep. The number was quoted in three places
-- here, the pre-flight estimate, which *divides its ETA by it* and was
therefore optimistic by a factor of two on every run the tool has ever
estimated, and the soft comparability key, which recorded a fact about the run
that was not true of it.

Serial is the right behaviour to keep. ``latency_p95_ms`` is a measurement of
the target, and two in-flight requests make some of it a measurement of
sweepeval's own queueing -- which is precisely why ``TimingPart.queue_ms``
exists and why ``total_ms`` excludes it. Raising this is a change to what the
latency family means, not a speedup, and it needs the queue subtraction
verified under real contention first.

So the constant moved to the truth rather than the behaviour moving to the
constant. Anything that reports concurrency must read it from here;
``tests/unit/test_concurrency_is_what_we_claim.py`` fails if what the tool
declares and what it does come apart again.
"""

MAX_ATTEMPTS = 4
BASE_DELAY_S = 1.0
MAX_DELAY_S = 60.0
CIRCUIT_BREAKER_THRESHOLD = 5


class BudgetExceeded(RuntimeError):
    """Raised when a request would take the run past its cap (§12.3)."""


@dataclass
class Budget:
    """Request and token caps. ``None`` means uncapped."""

    max_requests: int | None = None
    max_tokens: int | None = None
    requests: int = 0
    tokens: int = 0

    def would_exceed(self) -> bool:
        if self.max_requests is not None and self.requests >= self.max_requests:
            return True
        return self.max_tokens is not None and self.tokens >= self.max_tokens

    def record(self, *, requests: int = 1, tokens: int = 0) -> None:
        self.requests += requests
        self.tokens += tokens


@dataclass
class Governor:
    """Gates outbound requests for one target."""

    concurrency: int = DEFAULT_CONCURRENCY
    max_attempts: int = MAX_ATTEMPTS
    base_delay_s: float = BASE_DELAY_S
    max_delay_s: float = MAX_DELAY_S
    breaker_threshold: int = CIRCUIT_BREAKER_THRESHOLD
    seed: int = 0
    budget: Budget = field(default_factory=Budget)

    _semaphore: asyncio.Semaphore | None = field(default=None, init=False, repr=False)
    _rng: random.Random = field(init=False, repr=False)
    _consecutive_terminal: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _tripped: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    # --- concurrency -------------------------------------------------------

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Created lazily so a Governor can be built outside an event loop."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.concurrency)
        return self._semaphore

    # --- retry policy ------------------------------------------------------

    def should_retry(self, error_class: ErrorClass, attempt: int) -> bool:
        return error_class is ErrorClass.retryable and attempt < self.max_attempts

    def delay_for(self, attempt: int, retry_after: float | None = None) -> float:
        """Seconds to wait before ``attempt`` + 1.

        ``Retry-After`` always wins when the server sends it: the server knows
        its own capacity, and ignoring the header is how a client earns a
        longer ban.
        """
        if retry_after is not None:
            return min(retry_after, self.max_delay_s)
        ceiling = min(self.base_delay_s * (2 ** (attempt - 1)), self.max_delay_s)
        return self._rng.uniform(0.0, ceiling)

    # --- circuit breaker ---------------------------------------------------

    def record_outcome(self, config_id: str, error_class: ErrorClass) -> None:
        if error_class is ErrorClass.terminal:
            count = self._consecutive_terminal.get(config_id, 0) + 1
            self._consecutive_terminal[config_id] = count
            if count >= self.breaker_threshold:
                self._tripped.add(config_id)
        else:
            # Any non-terminal outcome resets the run: the breaker is for a
            # config that is systematically broken, not one that saw a 400 in
            # the middle of otherwise healthy traffic.
            self._consecutive_terminal[config_id] = 0

    def is_tripped(self, config_id: str) -> bool:
        """True once this config is ``ERRORED``; the sweep continues (§7)."""
        return config_id in self._tripped

    def consecutive_terminal(self, config_id: str) -> int:
        return self._consecutive_terminal.get(config_id, 0)

    # --- budget ------------------------------------------------------------

    def check_budget(self) -> None:
        if self.budget.would_exceed():
            raise BudgetExceeded(
                f"budget reached: {self.budget.requests} requests, "
                f"{self.budget.tokens} tokens"
            )

    def record_spend(self, *, requests: int = 1, tokens: int = 0) -> None:
        self.budget.record(requests=requests, tokens=tokens)


def parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header. Seconds form only; HTTP-date is ignored."""
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    return seconds if seconds >= 0 else None
