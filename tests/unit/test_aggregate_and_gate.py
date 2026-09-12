"""Aggregation and the gate rule (spec §13, §16)."""

from __future__ import annotations

import pytest

from sweepeval.schema.metric import CIMethod, Estimand, Flag
from sweepeval.schema.objective import Objective
from sweepeval.schema.observation import Observation, Verdict
from sweepeval.stats.aggregate import (
    LOW_COVERAGE_THRESHOLD,
    aggregate_metric,
    build_cluster_table,
    coverage_of,
)
from sweepeval.stats.diff import ExitCode, gate_metrics


def _obs(unit: str, verdict: Verdict, run: int = 0, **over: object) -> Observation:
    kwargs: dict[str, object] = dict(
        ts="t", run_id="r1", config_id="c1", unit_id=unit, run_idx=run,
        scorer="security", scorer_version=1, metric="security_pass_rate",
        family="security", layer="generic", verdict=verdict,
    )
    if verdict in (Verdict.UNSCORABLE, Verdict.SKIPPED):
        kwargs["reason"] = "refused"
    kwargs.update(over)
    return Observation(**kwargs)  # type: ignore[arg-type]


# --- clustering ------------------------------------------------------------


def test_runs_are_averaged_within_a_probe_before_it_becomes_a_cluster() -> None:
    """Probe x run as independent trials would inflate n by the run count.

    At temp=0 the three runs are often byte-identical, so the true information
    is one probe, not three, and treating them as three understates every
    interval.
    """
    rows = [
        _obs("u1", Verdict.PASS, 0), _obs("u1", Verdict.PASS, 1),
        _obs("u1", Verdict.FAIL, 2),
    ]
    table = build_cluster_table(rows, metric="security_pass_rate", config_id="c1")
    assert list(table.values) == ["u1"], "one cluster, not three"
    assert table.values["u1"] == pytest.approx(2 / 3)


def test_each_probe_is_its_own_cluster() -> None:
    rows = [_obs(f"u{i}", Verdict.PASS) for i in range(5)]
    table = build_cluster_table(rows, metric="security_pass_rate", config_id="c1")
    assert len(table.values) == 5


def test_observations_from_other_configs_are_ignored() -> None:
    rows = [_obs("u1", Verdict.PASS), _obs("u2", Verdict.FAIL, config_id="c2")]
    table = build_cluster_table(rows, metric="security_pass_rate", config_id="c1")
    assert list(table.values) == ["u1"]


def test_a_continuous_value_is_used_directly() -> None:
    rows = [_obs("u1", Verdict.PASS, value=420.0, metric="latency_ms")]
    table = build_cluster_table(rows, metric="latency_ms", config_id="c1")
    assert table.values["u1"] == 420.0


# --- exclusions (§11.8) ----------------------------------------------------


def test_unscorable_trials_leave_both_numerator_and_denominator() -> None:
    """A refused determinism trial is not a failed one."""
    rows = [
        _obs("u1", Verdict.PASS), _obs("u2", Verdict.UNSCORABLE),
        _obs("u3", Verdict.PASS),
    ]
    table = build_cluster_table(rows, metric="security_pass_rate", config_id="c1")
    assert set(table.values) == {"u1", "u3"}
    assert all(v == 1.0 for v in table.values.values())


def test_a_cluster_whose_every_trial_was_excluded_is_absent_not_zero() -> None:
    """A zero would read as a failure, which is the opposite of unscorable."""
    rows = [_obs("u1", Verdict.PASS), _obs("u2", Verdict.UNSCORABLE)]
    table = build_cluster_table(rows, metric="security_pass_rate", config_id="c1")
    assert "u2" not in table.values


def test_skipped_trials_are_counted_separately_from_unscorable() -> None:
    rows = [
        _obs("u1", Verdict.PASS), _obs("u2", Verdict.UNSCORABLE),
        _obs("u3", Verdict.SKIPPED),
    ]
    table = build_cluster_table(rows, metric="security_pass_rate", config_id="c1")
    assert (table.coverage.scored, table.coverage.unscorable, table.coverage.skipped) == (1, 1, 1)


def test_exclusion_reasons_are_kept_for_the_report() -> None:
    rows = [_obs("u1", Verdict.UNSCORABLE)]
    table = build_cluster_table(rows, metric="security_pass_rate", config_id="c1")
    assert "refused" in table.coverage.reasons


# --- coverage flagging -----------------------------------------------------


def test_heavy_exclusion_flags_low_coverage() -> None:
    """§11.8: over 30% unscorable and the number stops meaning what it says."""
    rows = [_obs("u1", Verdict.PASS)] + [
        _obs(f"x{i}", Verdict.UNSCORABLE) for i in range(3)
    ]
    table = build_cluster_table(rows, metric="security_pass_rate", config_id="c1")
    assert table.coverage.unscorable_fraction > LOW_COVERAGE_THRESHOLD
    assert Flag.LOW_COVERAGE in aggregate_metric(table, seed=1).flags


def test_light_exclusion_does_not_flag() -> None:
    rows = [_obs(f"u{i}", Verdict.PASS) for i in range(20)] + [
        _obs("x", Verdict.UNSCORABLE)
    ]
    table = build_cluster_table(rows, metric="security_pass_rate", config_id="c1")
    assert Flag.LOW_COVERAGE not in aggregate_metric(table, seed=1).flags


def test_coverage_of_reports_per_config() -> None:
    rows = [
        _obs("u1", Verdict.PASS),
        _obs("u2", Verdict.UNSCORABLE),
        _obs("u3", Verdict.PASS, config_id="c2"),
    ]
    report = coverage_of(rows, family="security")
    assert report["c1"].scored == 1
    assert report["c1"].unscorable == 1
    assert report["c2"].scored == 1


# --- aggregation -----------------------------------------------------------


def test_a_full_pass_aggregates_to_one_with_an_interval() -> None:
    rows = [_obs(f"u{i}", Verdict.PASS) for i in range(24)]
    table = build_cluster_table(rows, metric="security_pass_rate", config_id="c1")
    value = aggregate_metric(table, seed=1)
    assert value.point == 1.0
    assert value.lo is not None and value.lo < 1.0, "boundary correction applies"


def test_an_empty_table_declines_an_interval_rather_than_reporting_zero() -> None:
    table = build_cluster_table([], metric="security_pass_rate", config_id="c1")
    value = aggregate_metric(table, seed=1)
    assert value.method is CIMethod.none
    assert Flag.NO_VALID_INTERVAL in value.flags


def test_the_quick_profile_is_marked_indicative_not_invalid() -> None:
    """§6.4: wide is a different state from absent."""
    rows = [_obs(f"u{i}", Verdict.PASS) for i in range(10)]
    table = build_cluster_table(rows, metric="security_pass_rate", config_id="c1")
    value = aggregate_metric(table, seed=1, indicative=True)
    assert Flag.INDICATIVE in value.flags
    assert Flag.NO_VALID_INTERVAL not in value.flags
    assert value.lo is not None


def test_depth_stratification_is_carried_into_the_bootstrap() -> None:
    rows = [
        _obs(f"c{i}", Verdict.PASS, metric="retention", family="context",
             depth=[3, 8, 15][i % 3])
        for i in range(12)
    ]
    table = build_cluster_table(
        rows, metric="retention", config_id="c1", stratify_by_depth=True
    )
    assert set(table.strata.values()) == {"d3", "d8", "d15"}


def test_the_estimand_is_recorded() -> None:
    rows = [_obs(f"u{i}", Verdict.PASS) for i in range(12)]
    table = build_cluster_table(rows, metric="security_pass_rate", config_id="c1")
    value = aggregate_metric(table, estimand=Estimand.generalization, seed=1)
    assert value.estimand is Estimand.generalization


# --- the gate (§16) --------------------------------------------------------


def _objective(name: str, direction: str = "maximize", min_effect: float = 0.05) -> Objective:
    return Objective(
        id=name, display_label=name, direction=direction,  # type: ignore[arg-type]
        family="test", cluster_key="probe", min_effect=min_effect,
        min_effect_kind="absolute", weighting="test",
    )


def _clusters(value: float, n: int = 24) -> dict[str, float]:
    return {f"p{i}": value for i in range(n)}


def test_an_unchanged_run_passes() -> None:
    obj = _objective("security_pass_rate")
    verdict = gate_metrics(
        [obj], {"security_pass_rate": _clusters(0.9)},
        {"security_pass_rate": _clusters(0.9)}, seed=1,
    )
    assert verdict.ok
    assert verdict.exit_code is ExitCode.PASS


def test_a_large_regression_fires() -> None:
    obj = _objective("security_pass_rate")
    verdict = gate_metrics(
        [obj], {"security_pass_rate": _clusters(0.95)},
        {"security_pass_rate": _clusters(0.55)}, seed=1,
    )
    assert not verdict.ok
    assert verdict.exit_code is ExitCode.REGRESSION
    assert verdict.regressions[0].metric == "security_pass_rate"


def test_a_movement_below_the_practical_margin_does_not_fire() -> None:
    """§13.6: significance alone must never fail a build."""
    obj = _objective("security_pass_rate", min_effect=0.10)
    verdict = gate_metrics(
        [obj], {"security_pass_rate": _clusters(0.90, 200)},
        {"security_pass_rate": _clusters(0.87, 200)}, seed=1,
    )
    assert verdict.ok, verdict.explain()
    assert "practical-effect margin" in verdict.diffs[0].reason


def test_an_improvement_never_fires() -> None:
    obj = _objective("security_pass_rate")
    verdict = gate_metrics(
        [obj], {"security_pass_rate": _clusters(0.60)},
        {"security_pass_rate": _clusters(0.95)}, seed=1,
    )
    assert verdict.ok
    assert verdict.diffs[0].improved


def test_direction_is_respected_for_minimised_metrics() -> None:
    """Latency going up is a regression; going down is not."""
    obj = _objective("latency_p95_ms", direction="minimize", min_effect=10.0)
    worse = gate_metrics(
        [obj], {"latency_p95_ms": _clusters(300.0)},
        {"latency_p95_ms": _clusters(900.0)}, seed=1,
    )
    better = gate_metrics(
        [obj], {"latency_p95_ms": _clusters(900.0)},
        {"latency_p95_ms": _clusters(300.0)}, seed=1,
    )
    assert not worse.ok
    assert better.ok


def test_a_hard_fail_fails_regardless_of_the_baseline() -> None:
    """§16: confirmed security hard-fails bypass the comparison entirely."""
    obj = _objective("security_pass_rate")
    verdict = gate_metrics(
        [obj], {"security_pass_rate": _clusters(0.9)},
        {"security_pass_rate": _clusters(0.9)},
        hard_fails=("sec.exfiltration.direct.v1",), seed=1,
    )
    assert not verdict.ok
    assert verdict.hard_fails


def test_gate_on_restricts_which_metrics_can_fail() -> None:
    objs = [_objective("security_pass_rate"), _objective("latency_p95_ms", "minimize", 10.0)]
    clusters_base = {"security_pass_rate": _clusters(0.9), "latency_p95_ms": _clusters(300.0)}
    clusters_now = {"security_pass_rate": _clusters(0.9), "latency_p95_ms": _clusters(900.0)}

    ungated = gate_metrics(objs, clusters_base, clusters_now, seed=1)
    gated = gate_metrics(
        objs, clusters_base, clusters_now, gate_on=["security_pass_rate"], seed=1
    )
    assert not ungated.ok
    assert gated.ok, "latency excluded from the default gate (§16)"


def test_holm_corrects_across_gated_metrics() -> None:
    """Six objectives is six chances at a false fire."""
    objs = [_objective(f"m{i}") for i in range(6)]
    base = {o.id: _clusters(0.90) for o in objs}
    now = {o.id: _clusters(0.90) for o in objs}
    verdict = gate_metrics(objs, base, now, seed=1)
    assert verdict.ok
    for diff in verdict.diffs:
        assert diff.adjusted_threshold is not None
        assert diff.adjusted_threshold <= 0.05


def test_a_min_effect_override_is_honoured() -> None:
    obj = _objective("security_pass_rate", min_effect=0.01)
    strict = gate_metrics(
        [obj], {"security_pass_rate": _clusters(0.90, 200)},
        {"security_pass_rate": _clusters(0.85, 200)}, seed=1,
    )
    lenient = gate_metrics(
        [obj], {"security_pass_rate": _clusters(0.90, 200)},
        {"security_pass_rate": _clusters(0.85, 200)},
        min_effect_overrides={"security_pass_rate": 0.20}, seed=1,
    )
    assert not strict.ok
    assert lenient.ok


def test_the_explanation_shows_both_points_and_the_verdict() -> None:
    obj = _objective("security_pass_rate")
    verdict = gate_metrics(
        [obj], {"security_pass_rate": _clusters(0.95)},
        {"security_pass_rate": _clusters(0.55)}, seed=1,
    )
    text = verdict.explain()
    assert "security_pass_rate" in text
    assert "REGRESSION" in text
    assert "exit 1" in text


def test_exit_codes_match_the_spec() -> None:
    assert ExitCode.PASS == 0
    assert ExitCode.REGRESSION == 1
    assert ExitCode.COMPARABILITY_REFUSED == 2
    assert ExitCode.USAGE_ERROR == 3
