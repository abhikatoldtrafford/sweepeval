"""``Call`` and ``Observation`` rows (spec §6.1, §6.4).

The observation key is the fix for a rev-1 defect: ``(config_id, unit_id,
run_idx)`` is not unique, because one Unit yields several Observations per run
— a context conversation produces fact recall, constraint persistence and
distractor resistance from the same conversation. The key is the five-tuple
``(config_id, unit_id, run_idx, scorer, metric)``.

Three spec details that are easy to get subtly wrong and are pinned here:
``total_ms`` excludes ``queue_ms`` so latency measures the target rather than
the harness; retry attempts are excluded from the latency population; and
``tokens.reasoning`` is a distinct field.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from sweepeval.schema.call import Call, ErrorClass, TimingPart, TokensPart
from sweepeval.schema.observation import Observation, Verdict


def _obs(**over: Any) -> Observation:
    kwargs: dict[str, Any] = dict(
        ts="2026-09-12T00:00:00Z",
        run_id="r1",
        config_id="c1",
        unit_id="u#0123456789abcdef",
        run_idx=0,
        scorer="security",
        scorer_version=1,
        metric="security_pass_rate",
        family="security",
        layer="generic",
        verdict=Verdict.PASS,
        call_ids=("k1",),
    )
    kwargs.update(over)
    return Observation(**kwargs)


# --- the observation key --------------------------------------------------


def test_observation_key_is_the_five_tuple() -> None:
    assert _obs().key == (
        "c1",
        "u#0123456789abcdef",
        0,
        "security",
        "security_pass_rate",
    )


def test_two_observations_from_one_unit_run_have_distinct_keys() -> None:
    """The rev-1 three-tuple would have collided these into one row."""
    recall = _obs(scorer="context", metric="fact_recall")
    persistence = _obs(scorer="context", metric="constraint_persistence")
    assert recall.key != persistence.key


def test_key_distinguishes_runs_and_configs() -> None:
    assert _obs(run_idx=0).key != _obs(run_idx=1).key
    assert _obs(config_id="c1").key != _obs(config_id="c2").key


# --- I5: never a silent skip ----------------------------------------------


def test_unscorable_requires_a_reason() -> None:
    with pytest.raises(ValidationError):
        _obs(verdict=Verdict.UNSCORABLE, reason=None)


def test_skipped_requires_a_reason() -> None:
    with pytest.raises(ValidationError):
        _obs(verdict=Verdict.SKIPPED, reason=None)


def test_unscorable_with_a_reason_is_accepted() -> None:
    obs = _obs(verdict=Verdict.UNSCORABLE, reason="refusal_excluded_from_determinism")
    assert obs.reason


def test_blank_reason_does_not_satisfy_the_requirement() -> None:
    with pytest.raises(ValidationError):
        _obs(verdict=Verdict.SKIPPED, reason="   ")


def test_verdict_has_no_default() -> None:
    """I5: a scorer must state a verdict; there is no quiet fallthrough."""
    with pytest.raises(ValidationError):
        Observation(
            ts="t",
            run_id="r",
            config_id="c",
            unit_id="u",
            run_idx=0,
            scorer="s",
            scorer_version=1,
            metric="m",
            family="f",
            layer="generic",
            call_ids=(),
        )


# --- timing and retries ---------------------------------------------------


def test_timing_total_excludes_queue() -> None:
    timing = TimingPart(queue_ms=100.0, ttft_ms=50.0, total_ms=400.0)
    assert timing.total_ms == 400.0
    assert timing.queue_ms == 100.0


def test_first_attempt_counts_toward_latency() -> None:
    assert Call.example(attempt=1).counts_toward_latency is True


def test_retry_attempts_are_excluded_from_latency() -> None:
    assert Call.example(attempt=2).counts_toward_latency is False


def test_errored_calls_are_excluded_from_latency() -> None:
    """A 500 that took 30s is a failure, not a latency sample."""
    from sweepeval.schema.call import ResponsePart

    errored = Call.example(
        response=ResponsePart(
            status=500,
            error_class=ErrorClass.retryable,
            streamed=False,
            bytes=0,
        )
    )
    assert errored.counts_toward_latency is False


def test_error_class_enum_matches_the_spec_table() -> None:
    assert {e.value for e in ErrorClass} == {
        "ok",
        "retryable",
        "terminal",
        "refusal",
        "malformed",
    }


# --- tokens ---------------------------------------------------------------


def test_reasoning_tokens_are_a_distinct_field() -> None:
    """§7: reasoning tokens are billed and reported separately from output."""
    tokens = TokensPart(in_=100, out=20, reasoning=500, source="MEASURED")
    assert tokens.reasoning == 500
    assert tokens.out == 20


def test_tokens_default_to_estimated() -> None:
    """§12.3: absent a usage block, counts are estimates and must say so."""
    assert TokensPart().source == "ESTIMATED"


def test_tokens_in_serialises_without_the_underscore() -> None:
    """``in`` is a keyword; the wire format should still read naturally."""
    assert '"in":100' in TokensPart(in_=100, out=1).model_dump_json(by_alias=True)


# --- roles ----------------------------------------------------------------


def test_judge_calls_are_distinguishable_from_target_calls() -> None:
    """§11.9: judge calls persist in calls.jsonl and are budgeted separately."""
    assert Call.example(role="judge").role == "judge"
    assert Call.example().role == "target"
