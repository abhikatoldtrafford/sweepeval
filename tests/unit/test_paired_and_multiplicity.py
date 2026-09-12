"""Paired comparison and multiplicity (spec §13.3, §13.5). I2 load-bearing."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from sweepeval.schema.metric import CIMethod, Flag
from sweepeval.stats.multiplicity import (
    ObjectiveComparison,
    holm,
    holm_thresholds,
    pair_p_value,
)
from sweepeval.stats.paired import paired_difference


def _mean(values: dict[str, float]):
    def statistic(drawn: Sequence[str]) -> float:
        return sum(values[c] for c in drawn) / len(drawn)

    return statistic


# --- the pairing is what buys the power -----------------------------------


def test_pairing_cancels_probe_difficulty() -> None:
    """The point of the matched block.

    Probes vary hugely in difficulty, but b beats a by a constant margin on
    every one. Unpaired, the probe-to-probe spread swamps the effect; paired,
    the difference is exact and the interval is tight.
    """
    ids = [f"p{i}" for i in range(24)]
    hard = {cid: (i % 10) / 10.0 for i, cid in enumerate(ids)}
    a = dict(hard)
    b = {cid: min(1.0, v + 0.08) for cid, v in hard.items()}

    result = paired_difference(ids, _mean(a), _mean(b), seed=1)
    assert result.difference == pytest.approx(0.08, abs=0.02)
    assert (result.hi - result.lo) < 0.02, "pairing should leave almost no spread"


def test_no_difference_gives_an_interval_spanning_zero() -> None:
    ids = [f"p{i}" for i in range(24)]
    values = {cid: (i % 7) / 7.0 for i, cid in enumerate(ids)}
    result = paired_difference(ids, _mean(values), _mean(values), seed=1)
    assert result.lo <= 0.0 <= result.hi
    assert result.p_superior > 0.05


def test_a_real_advantage_produces_a_small_tail_area() -> None:
    ids = [f"p{i}" for i in range(24)]
    a = {cid: 0.5 for cid in ids}
    b = {cid: 0.9 for cid in ids}
    result = paired_difference(ids, _mean(a), _mean(b), seed=1)
    assert result.p_superior < 0.01
    assert result.p_non_inferior < 0.01, "b is ahead, so it is certainly not behind"


def test_direction_orients_the_tails() -> None:
    """For latency, lower is better, and the caller should not have to remember."""
    ids = [f"p{i}" for i in range(24)]
    slow = {cid: 900.0 for cid in ids}
    fast = {cid: 300.0 for cid in ids}
    result = paired_difference(ids, _mean(slow), _mean(fast), direction="minimize", seed=1)
    assert result.difference > 0, "sign-oriented: positive always means b is ahead"
    assert result.p_superior < 0.01, "b is faster, so b is better"


def test_results_are_reproducible_for_a_fixed_seed() -> None:
    ids = [f"p{i}" for i in range(20)]
    a = {cid: (i % 3) / 3.0 for i, cid in enumerate(ids)}
    b = {cid: (i % 4) / 4.0 for i, cid in enumerate(ids)}
    first = paired_difference(ids, _mean(a), _mean(b), seed=9)
    second = paired_difference(ids, _mean(a), _mean(b), seed=9)
    assert (first.lo, first.hi, first.p_superior) == (
        second.lo, second.hi, second.p_superior
    )


def test_stratified_resampling_is_used_when_given() -> None:
    ids = [f"c{i}" for i in range(9)]
    depths = {cid: f"d{i // 3}" for i, cid in enumerate(ids)}
    values = {cid: float(i) / 9 for i, cid in enumerate(ids)}
    result = paired_difference(ids, _mean(values), _mean(values), seed=1, strata=depths)
    assert result.evidence is not None
    assert result.evidence["stratified"] is True


def test_below_the_floor_is_flagged_low_n() -> None:
    ids = [f"p{i}" for i in range(5)]
    a = {cid: 0.5 for cid in ids}
    b = {cid: 0.7 for cid in ids}
    result = paired_difference(ids, _mean(a), _mean(b), seed=1, bounded=True)
    assert Flag.LOW_N in result.flags


def test_below_three_clusters_refuses_to_compare() -> None:
    """A wide interval would still license a domination claim; p=1 refuses to."""
    ids = ["p0", "p1"]
    a = {cid: 0.1 for cid in ids}
    b = {cid: 0.9 for cid in ids}
    result = paired_difference(ids, _mean(a), _mean(b), seed=1)
    assert result.method is CIMethod.none
    assert result.p_superior == 1.0
    assert result.p_non_inferior == 1.0
    assert Flag.NO_VALID_INTERVAL in result.flags


# --- the two-half construction (§13.5) ------------------------------------


def _cmp(name: str, diff: float, p_sup: float, p_ni: float) -> ObjectiveComparison:
    return ObjectiveComparison(
        objective=name, difference=diff, min_effect=0.02,
        p_superior=p_sup, p_non_inferior=p_ni,
    )


def test_domination_needs_both_halves() -> None:
    """Better on one, non-inferior on all."""
    p_pair, p_ni, p_sup = pair_p_value(
        [_cmp("security", 0.20, 0.001, 0.001), _cmp("latency", 0.00, 0.90, 0.001)]
    )
    assert p_ni == pytest.approx(0.001)
    assert p_sup == pytest.approx(0.002)
    assert p_pair == pytest.approx(0.002)


def test_being_worse_on_one_objective_blocks_domination() -> None:
    """The non-inferiority half is an IUT: the worst objective governs."""
    p_pair, p_ni, _ = pair_p_value(
        [_cmp("security", 0.20, 0.001, 0.001), _cmp("latency", -0.30, 0.99, 0.85)]
    )
    assert p_ni == pytest.approx(0.85)
    assert p_pair == pytest.approx(0.85)


def test_being_merely_tied_everywhere_blocks_domination() -> None:
    """Non-inferior on all but superior on none is a tie, not a win."""
    p_pair, _, p_sup = pair_p_value(
        [_cmp("security", 0.0, 0.50, 0.01), _cmp("latency", 0.0, 0.50, 0.01)]
    )
    assert p_sup == pytest.approx(1.0)
    assert p_pair == pytest.approx(1.0)


def test_the_superiority_union_is_bonferroni_corrected() -> None:
    """Six chances to find one win is six chances to find a false one."""
    six = [_cmp(f"m{i}", 0.1, 0.02, 0.001) for i in range(6)]
    _, _, p_sup = pair_p_value(six)
    assert p_sup == pytest.approx(0.12)


def test_the_non_inferiority_conjunction_needs_no_correction() -> None:
    """An intersection-union test is valid at face value — free rigour."""
    many = [_cmp(f"m{i}", 0.1, 0.001, 0.04) for i in range(6)]
    _, p_ni, _ = pair_p_value(many)
    assert p_ni == pytest.approx(0.04), "max, not 6 x 0.04"


def test_no_objectives_never_dominates() -> None:
    assert pair_p_value([]) == (1.0, 1.0, 1.0)


# --- Holm ------------------------------------------------------------------


def test_holm_rejects_the_smallest_p_first() -> None:
    result = holm({"a": 0.001, "b": 0.04, "c": 0.9}, alpha=0.05)
    assert result["a"] is True
    assert result["c"] is False


def test_holm_is_stricter_than_an_uncorrected_test() -> None:
    """0.04 passes alone and must not pass among ten."""
    alone = holm({"a": 0.04}, alpha=0.05)
    among = holm({f"p{i}": 0.04 for i in range(10)}, alpha=0.05)
    assert alone["a"] is True
    assert not any(among.values())


def test_holm_steps_down_and_stops_at_the_first_failure() -> None:
    """Everything after a failure is retained regardless of its own p-value."""
    result = holm({"a": 0.001, "b": 0.9, "c": 0.002}, alpha=0.05)
    assert result["a"] is True
    assert result["c"] is True
    assert result["b"] is False


def test_holm_thresholds_are_reported_for_the_prune_log() -> None:
    thresholds = holm_thresholds({"a": 0.001, "b": 0.04, "c": 0.9}, alpha=0.05)
    assert thresholds["a"] == pytest.approx(0.05 / 3)
    assert thresholds["c"] == pytest.approx(0.05)


def test_holm_on_an_empty_family_is_empty() -> None:
    assert holm({}) == {}


def test_holm_is_deterministic_under_ties() -> None:
    """Report ordering must not depend on dict insertion order."""
    first = holm({"b": 0.01, "a": 0.01}, alpha=0.05)
    second = holm({"a": 0.01, "b": 0.01}, alpha=0.05)
    assert first == second


# --- the inverted-semantics bug, pinned -----------------------------------


def test_a_config_that_is_worse_is_not_reported_as_non_inferior() -> None:
    """The bug the I2 simulation caught.

    An earlier construction derived non-inferiority from a tail area whose
    sense was inverted, so a config that was WORST on an objective scored as
    definitively non-inferior on it — and a genuine trade-off pair was declared
    dominated, which is a correctness failure rather than a statistical one.
    """
    ids = [f"p{i}" for i in range(24)]
    good = {cid: 0.95 for cid in ids}
    bad = {cid: 0.40 for cid in ids}

    result = paired_difference(ids, _mean(good), _mean(bad), margin=0.02, seed=1)
    assert result.difference < 0, "b is worse"
    assert result.p_non_inferior > 0.9, "and must not read as non-inferior"
    assert result.p_superior > 0.9, "nor as superior"


def test_the_margin_shifts_both_thresholds() -> None:
    """§13.6: a gap smaller than min_effect is neither a win nor a loss."""
    ids = [f"p{i}" for i in range(60)]
    a = {cid: 0.80 for cid in ids}
    b = {cid: 0.83 for cid in ids}

    generous = paired_difference(ids, _mean(a), _mean(b), margin=0.0, seed=1)
    strict = paired_difference(ids, _mean(a), _mean(b), margin=0.10, seed=1)

    assert generous.p_superior < 0.05, "a 3-point gap is significant at margin 0"
    assert strict.p_superior > 0.5, "and meaningless against a 10-point margin"
