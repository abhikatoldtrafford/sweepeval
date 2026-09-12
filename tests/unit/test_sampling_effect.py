"""The sampling-effect decision table (spec §9.1, D7)."""

from __future__ import annotations

import pytest

from sweepeval.capabilities.normalise import distinct_count, normalise, token_jaccard
from sweepeval.capabilities.sampling import (
    TOST_MARGIN,
    Verdict,
    decide_tier1,
    decide_tier2,
    dispersion,
    tost_equivalent,
)

# --- normalisation --------------------------------------------------------


def test_timestamps_are_masked() -> None:
    """Otherwise a target that stamps its replies looks infinitely creative."""
    a = "Done at 2026-09-12T10:00:00Z"
    b = "Done at 2026-09-12T10:00:01Z"
    assert normalise(a) == normalise(b)


def test_request_ids_are_masked() -> None:
    assert normalise("id chatcmpl-abc123def") == normalise("id chatcmpl-zzz999yyy")


def test_uuids_are_masked() -> None:
    a = "ref 123e4567-e89b-12d3-a456-426614174000"
    b = "ref 00000000-0000-0000-0000-000000000000"
    assert normalise(a) == normalise(b)


def test_whitespace_and_case_are_collapsed() -> None:
    assert normalise("  Hello   World \n") == normalise("hello world")


def test_genuinely_different_text_stays_different() -> None:
    assert normalise("red green blue") != normalise("cyan magenta yellow")


def test_distinct_count_uses_normalised_text() -> None:
    outputs = ["OK at 2026-09-12T10:00:00Z", "OK at 2026-09-12T11:00:00Z", "different"]
    assert distinct_count(outputs) == 2


def test_token_jaccard_bounds() -> None:
    assert token_jaccard("a b c", "a b c") == 1.0
    assert token_jaccard("a b", "c d") == 0.0
    assert 0 < token_jaccard("a b c", "a b d") < 1


# --- tier 1: the full decision table --------------------------------------


def _runs(*texts: str) -> list[str]:
    return list(texts)


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        # (distinct_low, distinct_high) -> verdict, per §9.1's table
        (["a", "a", "a"], ["x", "y", "z"], Verdict.EFFECTIVE),   # 1 | 3
        (["a", "a", "a"], ["x", "y", "y"], Verdict.ESCALATE),    # 1 | 2
        (["a", "a", "a"], ["x", "x", "x"], Verdict.ESCALATE),    # 1 | 1
        (["a", "b", "b"], ["x", "y", "z"], Verdict.EFFECTIVE),   # 2 | 3
        (["a", "b", "b"], ["x", "y", "y"], Verdict.ESCALATE),    # 2 | 2
        (["a", "b", "b"], ["x", "x", "x"], Verdict.ESCALATE),    # 2 | 1
        (["a", "b", "c"], ["x", "y", "z"], Verdict.ESCALATE),    # 3 | any
        (["a", "b", "c"], ["x", "x", "x"], Verdict.ESCALATE),    # 3 | any
    ],
)
def test_every_row_of_the_decision_table(
    low: list[str], high: list[str], expected: Verdict
) -> None:
    assert decide_tier1([low], [high]).verdict is expected


def test_the_nondeterministic_at_temp0_row_escalates() -> None:
    """The row rev 1 had no cell for.

    MoE routing, batching and GPU nondeterminism make temp=0 non-deterministic
    on most hosted endpoints. Distinct counts cannot separate the settings
    there, so only the dispersion test can.
    """
    verdict = decide_tier1([["a", "b", "c"]], [["x", "y", "z"]])
    assert verdict.verdict is Verdict.ESCALATE
    assert verdict.evidence["per_prompt_distinct"][0]["low"] == 3


def test_distinctness_is_per_prompt_not_pooled() -> None:
    """Pooled, two different prompts trivially give two distinct outputs, and a
    single deviation would declare EFFECTIVE."""
    low = [["a", "a", "a"], ["b", "b", "b"]]
    high = [["a", "a", "a"], ["b", "b", "b"]]
    assert decide_tier1(low, high).verdict is Verdict.ESCALATE


def test_one_prompt_showing_an_effect_settles_it() -> None:
    low = [["a", "a", "a"], ["b", "b", "b"]]
    high = [["a", "a", "a"], ["x", "y", "z"]]
    assert decide_tier1(low, high).verdict is Verdict.EFFECTIVE


def test_tier1_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError):
        decide_tier1([["a"]], [])


# --- dispersion -----------------------------------------------------------


def test_identical_runs_have_zero_dispersion() -> None:
    assert dispersion(["same text", "same text", "same text"]) == 0.0


def test_varied_runs_have_positive_dispersion() -> None:
    assert dispersion(["alpha beta", "gamma delta", "epsilon zeta"]) > 0.5


def test_dispersion_of_one_run_is_zero() -> None:
    assert dispersion(["only"]) == 0.0


# --- tier 2 verdicts ------------------------------------------------------


def test_a_clear_effect_reaches_effective() -> None:
    low = ["constant reply"] * 8
    high = [f"varied reply number {i} with distinct words {i}" for i in range(8)]
    verdict = decide_tier2("temperature", low, high, seed=1)
    assert verdict.verdict is Verdict.EFFECTIVE
    assert verdict.tier == 2


def test_a_genuinely_inert_parameter_reaches_inert_via_tost() -> None:
    """§9.1: INERT requires positive equivalence evidence, not a null result."""
    identical = ["the same reply every time"] * 8
    verdict = decide_tier2("top_p", identical, list(identical), seed=1)
    assert verdict.verdict is Verdict.INERT
    assert "TOST" in verdict.reason


def test_a_non_significant_difference_is_inconclusive_never_inert() -> None:
    """The absence-of-evidence error, refused.

    Both settings are noisy and similar. There is no significant difference and
    no equivalence either; removing the axis on that basis would silently
    truncate the experiment.
    """
    low = [f"reply {i} with several varying words here" for i in range(8)]
    high = [f"answer {i} containing other differing words" for i in range(8)]
    verdict = decide_tier2("temperature", low, high, seed=1)
    assert verdict.verdict is not Verdict.INERT


def test_inconclusive_axes_are_still_swept() -> None:
    """§9.1's asymmetry: an inert axis costs money, a missed one costs the
    experiment."""
    low = [f"reply {i} varying" for i in range(8)]
    high = [f"answer {i} differing" for i in range(8)]
    verdict = decide_tier2("temperature", low, high, seed=1)
    if verdict.verdict is Verdict.INCONCLUSIVE:
        assert verdict.swept is True


def test_effective_is_swept_and_inert_is_not() -> None:
    identical = ["same"] * 8
    inert = decide_tier2("top_p", identical, list(identical), seed=1)
    assert inert.swept is False

    high = [f"varied {i} words {i}" for i in range(8)]
    effective = decide_tier2("temperature", identical, high, seed=1)
    assert effective.swept is True


# --- TOST -----------------------------------------------------------------


def test_tost_declares_equivalence_for_identical_behaviour() -> None:
    identical = ["same reply"] * 8
    equivalent, evidence = tost_equivalent(identical, list(identical), seed=1)
    assert equivalent is True
    assert evidence["margin"] == TOST_MARGIN


def test_tost_refuses_equivalence_for_a_real_difference() -> None:
    low = ["constant"] * 8
    high = [f"quite different text number {i}" for i in range(8)]
    equivalent, _ = tost_equivalent(low, high, seed=1)
    assert equivalent is False


def test_tost_margin_is_the_documented_value() -> None:
    """§9.1 fixes it; it is surfaced, not configurable, in v0.1."""
    assert TOST_MARGIN == 0.05


def test_tier2_evidence_records_both_dispersions() -> None:
    verdict = decide_tier2("temperature", ["a"] * 8, ["b"] * 8, seed=1)
    assert "dispersion_low" in verdict.evidence
    assert "dispersion_high" in verdict.evidence
    assert "p_high_disperses_more" in verdict.evidence
