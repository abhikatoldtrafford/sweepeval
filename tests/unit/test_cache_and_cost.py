"""Cache detection and cost accounting (spec §12.7, §12.4, D41, D8)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sweepeval.execute.cache_detect import (
    AFFECTED_METRICS,
    MIN_DROP_MS,
    MIN_EVIDENCE_GROUPS,
    detect_cache,
    flag_affected,
)
from sweepeval.execute.cost import (
    DEGRADED_METRIC,
    Pricing,
    account,
    estimate_tokens,
)
from sweepeval.schema.call import (
    Call,
    ErrorClass,
    RequestPart,
    ResponsePart,
    TimingPart,
    TokensPart,
)
from sweepeval.schema.metric import CIMethod, Estimand, Flag, MetricValue


def _call(
    *,
    unit_id: str = "u1",
    run_idx: int = 0,
    turn_idx: int = 0,
    body: str = "a",
    response: str | None = "r",
    total_ms: float = 800.0,
    ttft_ms: float | None = None,
    attempt: int = 1,
    error: ErrorClass = ErrorClass.ok,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    reasoning: int | None = None,
    request_bytes: int = 400,
    response_bytes: int = 800,
) -> Call:
    return Call.example(
        unit_id=unit_id,
        run_idx=run_idx,
        turn_idx=turn_idx,
        attempt=attempt,
        request=RequestPart(
            shape="openai.chat_completions",
            params_hash="0" * 16,
            body_sha256=body * 64,
            bytes=request_bytes,
        ),
        response=ResponsePart(
            status=200,
            error_class=error,
            streamed=False,
            bytes=response_bytes,
            body_sha256=(response * 64) if response else None,
        ),
        timing=TimingPart(queue_ms=0.0, ttft_ms=ttft_ms, total_ms=total_ms),
        tokens=TokensPart(**{"in": tokens_in, "out": tokens_out, "reasoning": reasoning}),
    )


# --- cache detection (§12.7) ----------------------------------------------


def _cached_pair(unit_id: str, body: str) -> list[Call]:
    return [
        _call(unit_id=unit_id, run_idx=0, body=body, total_ms=800.0),
        _call(unit_id=unit_id, run_idx=1, body=body, total_ms=5.0),
    ]


def test_a_cache_is_detected_from_identical_bodies_answered_too_fast() -> None:
    calls = [*_cached_pair("u1", "a"), *_cached_pair("u2", "b")]
    verdict = detect_cache(calls)
    assert verdict.suspected
    assert "not trustworthy" in verdict.reason
    assert len(verdict.evidence) == 2


def test_a_deterministic_target_is_not_a_cache() -> None:
    """The whole point of the latency half of the rule.

    A target at temp=0 returns byte-identical responses; flagging that would
    make CACHE_SUSPECTED fire on exactly the result
    target_determinism_at_temp0 exists to measure.
    """
    calls = [
        _call(unit_id="u1", run_idx=0, body="a", total_ms=800.0),
        _call(unit_id="u1", run_idx=1, body="a", total_ms=790.0),
        _call(unit_id="u2", run_idx=0, body="b", total_ms=810.0),
        _call(unit_id="u2", run_idx=1, body="b", total_ms=805.0),
    ]
    verdict = detect_cache(calls)
    assert not verdict.suspected
    assert verdict.groups_examined == 2
    assert "none answered implausibly fast" in verdict.reason


def test_a_different_response_is_not_a_cache_hit() -> None:
    calls = [
        _call(unit_id="u1", run_idx=0, body="a", response="r", total_ms=800.0),
        _call(unit_id="u1", run_idx=1, body="a", response="s", total_ms=2.0),
        _call(unit_id="u2", run_idx=0, body="b", response="r", total_ms=800.0),
        _call(unit_id="u2", run_idx=1, body="b", response="s", total_ms=2.0),
    ]
    assert not detect_cache(calls).suspected


def test_one_fast_repeat_is_noise_not_a_pattern() -> None:
    calls = [
        *_cached_pair("u1", "a"),
        _call(unit_id="u2", run_idx=0, body="b", total_ms=800.0),
        _call(unit_id="u2", run_idx=1, body="b", total_ms=780.0),
    ]
    verdict = detect_cache(calls)
    assert not verdict.suspected
    assert f"{MIN_EVIDENCE_GROUPS}-body threshold" in verdict.reason


def test_a_fast_endpoint_is_not_flagged_by_proportional_noise() -> None:
    """A localhost or self-hosted target answers in single-digit ms.

    Without the absolute floor, 8ms -> 1ms clears the ratio and every local
    run reports a cache it does not have.
    """
    calls = [
        *(
            _call(unit_id=f"u{i}", run_idx=0, body=chr(97 + i), total_ms=8.0)
            for i in range(4)
        ),
        *(
            _call(unit_id=f"u{i}", run_idx=1, body=chr(97 + i), total_ms=1.0)
            for i in range(4)
        ),
    ]
    assert not detect_cache(calls).suspected
    assert MIN_DROP_MS > 7.0


def test_per_run_canaries_leave_nothing_to_compare() -> None:
    """Security units send a different body every run by design (§11.2)."""
    calls = [
        _call(unit_id="u1", run_idx=0, body="a", total_ms=800.0),
        _call(unit_id="u1", run_idx=1, body="b", total_ms=2.0),
    ]
    verdict = detect_cache(calls)
    assert not verdict.suspected
    assert "sent more than once" in verdict.reason


def test_ttft_is_never_compared_against_total() -> None:
    """Mixing the two manufactures a collapse out of a streaming difference.

    The first run did not report a TTFT and the second did. Falling back
    per-call rather than per-pair would compare run 0's 800ms *total* against
    run 1's 50ms *first token* and call a perfectly ordinary pair a cache.
    """
    calls = [
        _call(unit_id="u1", run_idx=0, body="a", total_ms=800.0, ttft_ms=None),
        _call(unit_id="u1", run_idx=1, body="a", total_ms=790.0, ttft_ms=50.0),
        _call(unit_id="u2", run_idx=0, body="b", total_ms=800.0, ttft_ms=None),
        _call(unit_id="u2", run_idx=1, body="b", total_ms=795.0, ttft_ms=50.0),
    ]
    assert not detect_cache(calls).suspected


def test_retries_and_errors_are_excluded() -> None:
    calls = [
        _call(unit_id="u1", run_idx=0, body="a", total_ms=800.0),
        _call(unit_id="u1", run_idx=1, body="a", total_ms=2.0, attempt=2),
        _call(unit_id="u2", run_idx=0, body="b", total_ms=800.0),
        _call(unit_id="u2", run_idx=1, body="b", total_ms=2.0, error=ErrorClass.retryable),
    ]
    assert not detect_cache(calls).suspected


def test_a_missing_response_hash_is_not_a_match() -> None:
    calls = [
        _call(unit_id="u1", run_idx=0, body="a", response=None, total_ms=800.0),
        _call(unit_id="u1", run_idx=1, body="a", response=None, total_ms=2.0),
        _call(unit_id="u2", run_idx=0, body="b", response=None, total_ms=800.0),
        _call(unit_id="u2", run_idx=1, body="b", response=None, total_ms=2.0),
    ]
    assert not detect_cache(calls).suspected


def _metric(point: float = 0.9) -> MetricValue:
    return MetricValue(
        point=point, lo=point - 0.1, hi=point + 0.05,
        method=CIMethod.cluster_bootstrap, n_clusters=10, alpha=0.05,
        estimand=Estimand.generalization,
    )


def test_the_flag_lands_on_determinism_and_latency_only() -> None:
    verdict = detect_cache([*_cached_pair("u1", "a"), *_cached_pair("u2", "b")])
    flagged = flag_affected(
        {
            "target_determinism_at_temp0": _metric(),
            "latency_ms": _metric(400.0),
            "security_pass_rate": _metric(),
        },
        verdict,
    )
    assert Flag.CACHE_SUSPECTED in flagged["target_determinism_at_temp0"].flags
    assert Flag.CACHE_SUSPECTED in flagged["latency_ms"].flags
    # A cached refusal is still a refusal.
    assert Flag.CACHE_SUSPECTED not in flagged["security_pass_rate"].flags
    assert "security_pass_rate" not in AFFECTED_METRICS


def test_no_cache_leaves_every_metric_untouched() -> None:
    metrics = {"latency_ms": _metric(400.0)}
    flagged = flag_affected(metrics, detect_cache([]))
    assert flagged["latency_ms"].flags == ()


# --- cost (§12.4, D8) ------------------------------------------------------


def test_reported_tokens_are_measured() -> None:
    calls = [
        _call(unit_id="u1", run_idx=r, tokens_in=100, tokens_out=50) for r in range(3)
    ]
    accounting = account(calls)
    assert accounting.source == "MEASURED"
    assert accounting.tokens_out == 150
    assert accounting.heuristic is None


def test_missing_usage_degrades_to_the_disclosed_heuristic() -> None:
    calls = [_call(unit_id="u1", run_idx=r) for r in range(3)]
    accounting = account(calls)
    assert accounting.source == "ESTIMATED"
    assert accounting.heuristic == "chars/4"


def test_partial_usage_is_estimated_not_half_measured() -> None:
    """A figure that is part measurement and part heuristic must not be
    labelled MEASURED."""
    calls = [
        _call(unit_id="u1", run_idx=0, tokens_out=50),
        _call(unit_id="u2", run_idx=0, tokens_out=50),
        _call(unit_id="u3", run_idx=0),
    ]
    assert account(calls).source == "ESTIMATED"


def test_no_pricing_degrades_the_metric_rather_than_inventing_a_price() -> None:
    calls = [_call(unit_id="u1", run_idx=r, tokens_out=100) for r in range(2)]
    accounting = account(calls)
    assert accounting.metric == DEGRADED_METRIC
    assert accounting.pricing_source == "none"
    assert accounting.degraded
    assert "no pricing supplied" in accounting.describe()


def test_user_pricing_produces_money_per_probe() -> None:
    calls = [_call(unit_id="u1", run_idx=r, tokens_in=1000, tokens_out=500) for r in range(2)]
    accounting = account(
        calls, pricing=Pricing(input_per_mtok=1.0, output_per_mtok=2.0)
    )
    assert accounting.metric == "cost_per_probe"
    assert accounting.currency == "USD"
    # 2 probes, 2000 in + 1000 out => (0.002 + 0.002) / 2
    assert accounting.value_per_probe == pytest.approx(0.002)


def test_reasoning_tokens_are_billed_as_output() -> None:
    """They are billed everywhere that reports them, and are not in the answer."""
    calls = [_call(unit_id="u1", run_idx=0, tokens_in=0, tokens_out=0, reasoning=1_000_000)]
    accounting = account(
        calls, pricing=Pricing(input_per_mtok=0.0, output_per_mtok=3.0)
    )
    assert accounting.value_per_probe == pytest.approx(3.0)


def test_cost_is_per_probe_not_per_call() -> None:
    """A depth-15 conversation is one probe and fifteen calls."""
    calls = [
        _call(unit_id="u1", run_idx=0, turn_idx=t, tokens_out=100) for t in range(15)
    ]
    accounting = account(calls)
    assert accounting.probes == 1
    assert accounting.value_per_probe == pytest.approx(1500.0)


def test_the_heuristic_is_the_one_the_manifest_names() -> None:
    assert estimate_tokens("a" * 400) == 100


def test_no_price_table_ships_in_the_repo() -> None:
    """D8. A stale bundled price is a confidently wrong dollar figure.

    The check is for a price *literal*, not for the word "price". Reading a
    number the user supplied is the supported path; writing one down is not.
    """
    import re

    src = Path(__file__).resolve().parents[2] / "src" / "sweepeval"
    literal = re.compile(r"per_mtok\s*[=:]\s*[0-9]")
    table = re.compile(r"(PRICES|PRICE_TABLE|price_table)")

    suspicious = []
    for path in [*src.rglob("*.py"), *src.rglob("*.yaml")]:
        text = path.read_text(encoding="utf-8")
        if literal.search(text):
            suspicious.append(f"{path.name}: a hardcoded price")
        if table.search(text):
            suspicious.append(f"{path.name}: a price table")
    assert not suspicious, suspicious


def test_the_price_literal_check_would_catch_one(tmp_path: Path) -> None:
    """A check you have not seen fail is not a check."""
    import re

    literal = re.compile(r"per_mtok\s*[=:]\s*[0-9]")
    assert literal.search('Pricing(input_per_mtok=0.15, output_per_mtok=0.60)')
    assert literal.search("  input_per_mtok: 2.50")
    assert not literal.search("input_per_mtok=float(value[\"input_per_mtok\"])")
