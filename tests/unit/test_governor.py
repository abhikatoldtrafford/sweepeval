"""Governor: concurrency, backoff, circuit breaker, budget (spec §7, D25)."""

from __future__ import annotations

import pytest

from sweepeval.http.governor import (
    BASE_DELAY_S,
    CIRCUIT_BREAKER_THRESHOLD,
    DEFAULT_CONCURRENCY,
    MAX_ATTEMPTS,
    MAX_DELAY_S,
    Budget,
    BudgetExceeded,
    Governor,
    parse_retry_after,
)
from sweepeval.schema.call import ErrorClass

# --- documented defaults --------------------------------------------------


def test_defaults_match_d25() -> None:
    assert DEFAULT_CONCURRENCY == 2
    assert MAX_ATTEMPTS == 4
    assert BASE_DELAY_S == 1.0
    assert MAX_DELAY_S == 60.0
    assert CIRCUIT_BREAKER_THRESHOLD == 5


# --- retry policy ---------------------------------------------------------


def test_only_retryable_errors_are_retried() -> None:
    gov = Governor()
    assert gov.should_retry(ErrorClass.retryable, attempt=1)
    assert not gov.should_retry(ErrorClass.terminal, attempt=1)
    assert not gov.should_retry(ErrorClass.refusal, attempt=1)
    assert not gov.should_retry(ErrorClass.malformed, attempt=1)


def test_retries_stop_at_the_attempt_cap() -> None:
    gov = Governor()
    assert gov.should_retry(ErrorClass.retryable, attempt=MAX_ATTEMPTS - 1)
    assert not gov.should_retry(ErrorClass.retryable, attempt=MAX_ATTEMPTS)


def test_backoff_ceiling_doubles_per_attempt() -> None:
    """Full jitter draws from [0, ceiling); the ceiling is what doubles."""
    gov = Governor(seed=1)
    for attempt, ceiling in ((1, 1.0), (2, 2.0), (3, 4.0), (4, 8.0)):
        draws = [gov.delay_for(attempt) for _ in range(200)]
        assert max(draws) <= ceiling
        assert max(draws) > ceiling / 2, "jitter should span most of the ceiling"


def test_backoff_is_capped() -> None:
    gov = Governor(seed=1)
    assert gov.delay_for(attempt=20) <= MAX_DELAY_S


def test_backoff_is_reproducible_for_a_fixed_seed() -> None:
    a = [Governor(seed=7).delay_for(2) for _ in range(1)]
    b = [Governor(seed=7).delay_for(2) for _ in range(1)]
    assert a == b


def test_retry_after_overrides_the_backoff_schedule() -> None:
    """The server knows its own capacity; ignoring the header earns a ban."""
    assert Governor(seed=1).delay_for(attempt=1, retry_after=3.0) == 3.0


def test_retry_after_is_still_capped() -> None:
    assert Governor().delay_for(attempt=1, retry_after=9999.0) == MAX_DELAY_S


@pytest.mark.parametrize(
    ("header", "expected"),
    [("3", 3.0), ("0", 0.0), ("2.5", 2.5), (None, None), ("", None), ("junk", None), ("-1", None)],
)
def test_retry_after_parsing(header: str | None, expected: float | None) -> None:
    assert parse_retry_after(header) == expected


# --- circuit breaker ------------------------------------------------------


def test_breaker_trips_on_the_fifth_consecutive_terminal_error() -> None:
    gov = Governor()
    for _ in range(CIRCUIT_BREAKER_THRESHOLD - 1):
        gov.record_outcome("c1", ErrorClass.terminal)
    assert not gov.is_tripped("c1")
    gov.record_outcome("c1", ErrorClass.terminal)
    assert gov.is_tripped("c1")


def test_a_success_resets_the_breaker_run() -> None:
    """The breaker is for a systematically broken config, not for a config
    that saw one 400 in the middle of healthy traffic."""
    gov = Governor()
    for _ in range(4):
        gov.record_outcome("c1", ErrorClass.terminal)
    gov.record_outcome("c1", ErrorClass.ok)
    assert gov.consecutive_terminal("c1") == 0
    for _ in range(4):
        gov.record_outcome("c1", ErrorClass.terminal)
    assert not gov.is_tripped("c1")


def test_a_refusal_resets_the_breaker_run() -> None:
    """A refusal is correct behaviour, not a fault."""
    gov = Governor()
    for _ in range(4):
        gov.record_outcome("c1", ErrorClass.terminal)
    gov.record_outcome("c1", ErrorClass.refusal)
    assert gov.consecutive_terminal("c1") == 0


def test_the_breaker_is_per_config_so_the_sweep_continues() -> None:
    gov = Governor()
    for _ in range(CIRCUIT_BREAKER_THRESHOLD):
        gov.record_outcome("c1", ErrorClass.terminal)
    assert gov.is_tripped("c1")
    assert not gov.is_tripped("c2")


# --- budget ---------------------------------------------------------------


def test_budget_is_uncapped_by_default() -> None:
    gov = Governor()
    for _ in range(1000):
        gov.record_spend(requests=1, tokens=1000)
    gov.check_budget()


def test_request_cap_is_enforced() -> None:
    gov = Governor(budget=Budget(max_requests=3))
    for _ in range(3):
        gov.check_budget()
        gov.record_spend()
    with pytest.raises(BudgetExceeded, match="3 requests"):
        gov.check_budget()


def test_token_cap_is_enforced() -> None:
    gov = Governor(budget=Budget(max_tokens=100))
    gov.record_spend(tokens=120)
    with pytest.raises(BudgetExceeded):
        gov.check_budget()


def test_spend_accumulates() -> None:
    gov = Governor()
    gov.record_spend(requests=2, tokens=50)
    gov.record_spend(requests=1, tokens=25)
    assert (gov.budget.requests, gov.budget.tokens) == (3, 75)


# --- concurrency ----------------------------------------------------------


@pytest.mark.asyncio
async def test_semaphore_admits_only_the_configured_concurrency() -> None:
    gov = Governor(concurrency=2)
    await gov.semaphore.acquire()
    await gov.semaphore.acquire()
    assert gov.semaphore.locked()
    gov.semaphore.release()
    assert not gov.semaphore.locked()


def test_a_governor_can_be_built_outside_an_event_loop() -> None:
    """Planning constructs one before asyncio.run is called."""
    assert Governor().concurrency == 2
