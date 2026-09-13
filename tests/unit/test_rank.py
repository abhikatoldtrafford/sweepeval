"""Ranking: constraints, coverage, clusters, correlation, prefer (spec §14)."""

from __future__ import annotations

import pytest

from sweepeval.execute.hard_fail import classify_hard_fails
from sweepeval.rank.cluster import cluster_ties, spread_of
from sweepeval.rank.constraints import (
    DEFAULT_CONSTRAINTS,
    Constraint,
    apply_constraints,
)
from sweepeval.rank.coverage import (
    BLOCK_THRESHOLD,
    EXCLUDE_THRESHOLD,
    coverage_matrix,
)
from sweepeval.rank.frontier import rank_configs
from sweepeval.rank.prefer import apply_preference, parse_preference
from sweepeval.report.frontier_json import FORBIDDEN_FIELDS, frontier_payload
from sweepeval.schema.metric import CIMethod, Estimand, Flag, MetricValue
from sweepeval.schema.objective import REGISTRY
from sweepeval.schema.observation import Observation, Verdict
from sweepeval.stats.correlation import correlation_matrix


def _mv(point: float, lo: float | None = None, hi: float | None = None, **kw):
    if lo is None:
        lo, hi = point - 0.05, point + 0.05
    return MetricValue(
        point=point, lo=lo, hi=hi, method=CIMethod.cluster_bootstrap,
        n_clusters=kw.pop("n_clusters", 12), alpha=0.05,
        estimand=Estimand.generalization, **kw,
    )


def _no_interval(point: float = 0.0) -> MetricValue:
    return MetricValue(
        point=point, method=CIMethod.none, n_clusters=0, alpha=0.05,
        estimand=Estimand.generalization, flags=(Flag.NO_VALID_INTERVAL,),
    )


# --- constraints (§14.1) ---------------------------------------------------


def test_error_rate_is_judged_on_the_favourable_bound_not_the_point() -> None:
    """A point-estimate threshold on a noisy rate is a coin flip.

    This config's point estimate is over the 0.05 limit and its interval says
    it could easily be under. It is not excluded.
    """
    metrics = {"c1": {"error_rate": _mv(0.08, lo=0.01, hi=0.15)}}
    report = apply_constraints(["c1"], metrics, [Constraint("error_rate", 0.05)])
    assert report.eligible == ("c1",)
    assert not report.violations


def test_a_config_that_could_not_be_within_the_limit_is_excluded() -> None:
    metrics = {"c1": {"error_rate": _mv(0.30, lo=0.20, hi=0.40)}}
    report = apply_constraints(["c1"], metrics, [Constraint("error_rate", 0.05)])
    assert not report.eligible
    assert report.violations[0].constraint.metric == "error_rate"
    assert "0.2" in report.violations[0].describe()


def test_hard_fails_are_a_census_not_an_estimate() -> None:
    """A leak that happened, happened. Running the count through the interval
    path would read a missing interval as 'not violated'."""
    report = apply_constraints(
        ["c1", "c2"],
        {"c1": {}, "c2": {}},
        DEFAULT_CONSTRAINTS,
        counts={"c1": {"security_hard_fails": 1.0}, "c2": {"security_hard_fails": 0.0}},
    )
    assert report.eligible == ("c2",)
    assert report.violations[0].config_id == "c1"
    assert "observed 1" in report.violations[0].describe()


def test_an_unmeasured_metric_does_not_exclude() -> None:
    """A measurement failure is not a verdict about the target (I5)."""
    metrics = {"c1": {"error_rate": _no_interval()}}
    report = apply_constraints(["c1"], metrics, [Constraint("error_rate", 0.05)])
    assert report.eligible == ("c1",)
    assert ("c1", "error_rate") in report.not_measured


# --- tied clusters (§14.6, D19) --------------------------------------------


def _chain_tie(a: str, b: str) -> bool:
    """A ties B, B ties C, A and C are significantly separated."""
    pair = frozenset({a, b})
    return pair in ({frozenset({"A", "B"}), frozenset({"B", "C"})})


def test_a_chain_never_lands_in_one_cluster() -> None:
    """Connected components would merge A, B and C. The tie relation is not
    transitive, and with six objectives components' modal outcome is one
    cluster containing everything."""
    clusters = cluster_ties(["A", "B", "C"], _chain_tie)
    members = [c.members for c in clusters]
    assert not any({"A", "C"} <= set(m) for m in members), members


def test_every_published_cluster_is_all_pairs_tied() -> None:
    clusters = cluster_ties(["A", "B", "C"], _chain_tie)
    for cluster in clusters:
        for a in cluster.members:
            for b in cluster.members:
                if a != b:
                    assert _chain_tie(a, b), (a, b)


def test_the_cover_is_stable_across_shuffled_inputs() -> None:
    """A reader comparing two runs needs the grouping to be stable."""
    orders = (
        ["A", "B", "C", "D"],
        ["D", "C", "B", "A"],
        ["C", "A", "D", "B"],
    )
    covers = {
        tuple(c.members for c in cluster_ties(order, _chain_tie)) for order in orders
    }
    assert len(covers) == 1, covers


def test_the_tie_break_prefers_fewer_clusters() -> None:
    everything_ties = {"A", "B", "C"}

    def tied(a: str, b: str) -> bool:
        return a in everything_ties and b in everything_ties

    clusters = cluster_ties(["A", "B", "C"], tied)
    assert len(clusters) == 1
    assert clusters[0].members == ("A", "B", "C")


def test_an_asymmetric_tie_fails_closed() -> None:
    """A relation that is not symmetric is a bug upstream; merging on it
    anyway would depend on which config the loop reached first."""

    def one_way(a: str, b: str) -> bool:
        return (a, b) == ("A", "B")

    clusters = cluster_ties(["A", "B"], one_way)
    assert len(clusters) == 2


def test_wide_spread_is_reported_despite_the_ties() -> None:
    """Wide spread with no significance is a signal to raise N."""
    cluster = cluster_ties(["A", "B"], lambda a, b: True)[0]
    spread_of(
        cluster,
        {"A": {"security_pass_rate": 0.60}, "B": {"security_pass_rate": 0.95}},
        ["security_pass_rate"],
    )
    assert cluster.spread["security_pass_rate"] == (0.60, 0.95)
    assert "security_pass_rate" in cluster.wide


def test_narrow_spread_is_not_called_wide() -> None:
    cluster = cluster_ties(["A", "B"], lambda a, b: True)[0]
    spread_of(
        cluster,
        {"A": {"security_pass_rate": 0.94}, "B": {"security_pass_rate": 0.95}},
        ["security_pass_rate"],
    )
    assert not cluster.wide


# --- coverage parity (§14.5) -----------------------------------------------


def _obs(config_id: str, unit: str, family: str, verdict: Verdict) -> Observation:
    return Observation(
        ts="2026-09-12T00:00:00Z", run_id="r", config_id=config_id, unit_id=unit,
        run_idx=0, scorer=family, scorer_version=1, metric=f"{family}_pass_rate",
        family=family, verdict=verdict, layer="generic",
        reason="x" if verdict in (Verdict.UNSCORABLE, Verdict.SKIPPED) else None,
    )


def test_a_coverage_gap_blocks_domination_for_that_pair() -> None:
    """A config that failed every depth-15 conversation cannot be dominated by
    one with full coverage: they were not scored on the same probes."""
    full = [_obs("full", f"u{i}", "context", Verdict.PASS) for i in range(10)]
    gappy = [
        _obs("gappy", f"u{i}", "context", Verdict.PASS if i < 7 else Verdict.UNSCORABLE)
        for i in range(10)
    ]
    matrix = coverage_matrix({"full": full, "gappy": gappy})
    assert matrix.divergence("full", "gappy", "context") == pytest.approx(0.3)
    assert matrix.blocks("full", "gappy", "context")
    assert BLOCK_THRESHOLD < 0.3


def test_equal_coverage_blocks_nothing() -> None:
    """Two configs with the same gap line up again and can be compared."""
    rows = {
        name: [
            _obs(name, f"u{i}", "context", Verdict.PASS if i < 7 else Verdict.UNSCORABLE)
            for i in range(10)
        ]
        for name in ("a", "b")
    }
    matrix = coverage_matrix(rows)
    assert not matrix.blocks("a", "b", "context")


def test_a_severe_gap_excludes_the_metric() -> None:
    full = [_obs("full", f"u{i}", "context", Verdict.PASS) for i in range(10)]
    broken = [_obs("broken", f"u{i}", "context", Verdict.UNSCORABLE) for i in range(10)]
    matrix = coverage_matrix({"full": full, "broken": broken})
    assert matrix.excludes("full", "broken", "context")
    assert EXCLUDE_THRESHOLD < 1.0


# --- hard fails (§11.2) ----------------------------------------------------


def _sec(unit: str, run_idx: int, verdict: Verdict, attack: str) -> Observation:
    return Observation(
        ts="2026-09-12T00:00:00Z", run_id="r", config_id="c1", unit_id=unit,
        run_idx=run_idx, scorer="security", scorer_version=1,
        metric="security_pass_rate", family="security", verdict=verdict,
        layer="generic", attack_class=attack,
        reason="x" if verdict in (Verdict.UNSCORABLE, Verdict.SKIPPED) else None,
    )


def test_a_leak_on_every_run_is_a_confirmed_hard_fail() -> None:
    from sweepeval.scorers.security import HARD_FAIL_CLASSES

    attack = sorted(HARD_FAIL_CLASSES)[0]
    report = classify_hard_fails(
        [_sec("u1", r, Verdict.FAIL, attack) for r in range(3)], config_id="c1"
    )
    assert report.count == 1
    assert report.confirmed[0].confirmed
    assert "3 of 3" in report.confirmed[0].reason


def test_an_intermittent_leak_is_suspected_not_confirmed() -> None:
    """Disqualifying a config on one draw is the point-estimate reasoning I3
    rejects everywhere else."""
    from sweepeval.scorers.security import HARD_FAIL_CLASSES

    attack = sorted(HARD_FAIL_CLASSES)[0]
    report = classify_hard_fails(
        [
            _sec("u1", 0, Verdict.FAIL, attack),
            _sec("u1", 1, Verdict.PASS, attack),
            _sec("u1", 2, Verdict.PASS, attack),
        ],
        config_id="c1",
    )
    assert report.count == 0
    assert report.suspected and "intermittent" in report.suspected[0].reason


def test_a_single_run_cannot_confirm() -> None:
    from sweepeval.scorers.security import HARD_FAIL_CLASSES

    attack = sorted(HARD_FAIL_CLASSES)[0]
    report = classify_hard_fails([_sec("u1", 0, Verdict.FAIL, attack)], config_id="c1")
    assert report.count == 0
    assert "nothing to confirm against" in report.suspected[0].reason


def test_a_low_severity_leak_is_not_a_hard_fail() -> None:
    """Severity drives hard-fail classification; it does not drive weighting."""
    report = classify_hard_fails(
        [_sec("u1", r, Verdict.FAIL, "not_a_hard_fail_class") for r in range(3)],
        config_id="c1",
    )
    assert report.count == 0
    assert not report.hard_fails


# --- correlation (§14.3) ---------------------------------------------------


def test_two_objectives_that_track_each_other_are_named() -> None:
    points = {
        f"c{i}": {"latency_p95_ms": float(i), "cost_per_probe": float(i) * 2}
        for i in range(5)
    }
    matrix = correlation_matrix(points, ["latency_p95_ms", "cost_per_probe"])
    high = matrix.high_pairs()
    assert high and high[0][2] == pytest.approx(1.0)


def test_a_constant_objective_is_excluded_rather_than_printed_as_nan() -> None:
    points = {
        f"c{i}": {"security_pass_rate": 1.0, "latency_p95_ms": float(i)}
        for i in range(5)
    }
    matrix = correlation_matrix(
        points, ["security_pass_rate", "latency_p95_ms"]
    )
    assert "security_pass_rate" not in matrix.objectives


def test_too_few_configs_says_so_rather_than_reporting_a_perfect_fit() -> None:
    points = {"a": {"x": 1.0, "y": 2.0}, "b": {"x": 2.0, "y": 4.0}}
    matrix = correlation_matrix(points, ["x", "y"])
    assert not matrix.computed
    assert "summarises nothing" in matrix.reason


# --- the frontier and --prefer (§14.4, §14.6) ------------------------------


def _sweep_fixture():
    """Three configs: one clean winner, one trade-off, one strictly worse."""
    objectives = [
        REGISTRY.get("security_pass_rate"),
        REGISTRY.get("latency_p95_ms"),
    ]
    units = [f"u{i}" for i in range(12)]

    clusters = {
        "fast": {
            "security_pass_rate": {u: 0.5 for u in units},
            "latency_p95_ms": {u: 100.0 for u in units},
        },
        "safe": {
            "security_pass_rate": {u: 1.0 for u in units},
            "latency_p95_ms": {u: 900.0 for u in units},
        },
        "bad": {
            "security_pass_rate": {u: 0.5 for u in units},
            "latency_p95_ms": {u: 900.0 for u in units},
        },
    }
    metrics = {
        "fast": {"security_pass_rate": _mv(0.5), "latency_p95_ms": _mv(100.0, 90.0, 110.0)},
        "safe": {
            "security_pass_rate": _mv(1.0, 0.95, 1.0),
            "latency_p95_ms": _mv(900.0, 880.0, 920.0),
        },
        "bad": {"security_pass_rate": _mv(0.5), "latency_p95_ms": _mv(900.0, 880.0, 920.0)},
    }
    return objectives, clusters, metrics


def test_a_strictly_worse_config_is_dominated() -> None:
    objectives, clusters, metrics = _sweep_fixture()
    result = rank_configs(
        ["fast", "safe", "bad"], metrics, clusters, objectives, constraints=()
    )
    assert "bad" in result.dominated
    assert set(result.frontier) == {"fast", "safe"}


def test_a_trade_off_pair_is_not_dominated_either_way() -> None:
    """I2: conservative by design. Neither is non-inferior on both."""
    objectives, clusters, metrics = _sweep_fixture()
    result = rank_configs(
        ["fast", "safe", "bad"], metrics, clusters, objectives, constraints=()
    )
    assert result.domination is not None
    assert not result.domination.dominates("fast", "safe")
    assert not result.domination.dominates("safe", "fast")


def test_prefer_lexicographic_names_one_config_and_what_it_concedes() -> None:
    objectives, clusters, metrics = _sweep_fixture()
    result = rank_configs(
        ["fast", "safe", "bad"], metrics, clusters, objectives, constraints=()
    )
    outcome = apply_preference(parse_preference("security,latency"), result, metrics)
    assert outcome.chosen == "safe"
    assert "latency_p95_ms" in outcome.concedes
    assert outcome.concedes["latency_p95_ms"] == ("fast",)


def test_prefer_reverses_with_the_priority() -> None:
    objectives, clusters, metrics = _sweep_fixture()
    result = rank_configs(
        ["fast", "safe", "bad"], metrics, clusters, objectives, constraints=()
    )
    outcome = apply_preference(parse_preference("latency,security"), result, metrics)
    assert outcome.chosen == "fast"


def test_prefer_constrained_excludes_and_says_which_bound() -> None:
    objectives, clusters, metrics = _sweep_fixture()
    result = rank_configs(
        ["fast", "safe", "bad"], metrics, clusters, objectives, constraints=()
    )
    outcome = apply_preference(
        parse_preference("maximize security_pass_rate subject to latency_p95_ms < 500"),
        result,
        metrics,
    )
    assert outcome.chosen == "fast"
    assert "safe" in outcome.excluded
    assert "latency_p95_ms < 500" in outcome.excluded["safe"]


def test_an_impossible_constraint_says_so_rather_than_picking_anyway() -> None:
    objectives, clusters, metrics = _sweep_fixture()
    result = rank_configs(
        ["fast", "safe", "bad"], metrics, clusters, objectives, constraints=()
    )
    outcome = apply_preference(
        parse_preference("maximize security_pass_rate subject to latency_p95_ms < 1"),
        result,
        metrics,
    )
    assert outcome.chosen is None
    assert "no configuration on the frontier satisfies" in outcome.reason


def test_an_unreadable_preference_shows_both_grammars() -> None:
    with pytest.raises(ValueError, match="Two forms are accepted"):
        parse_preference("make it good please")


# --- I1: nothing composite is stored --------------------------------------


def test_no_scalar_rank_field_appears_in_the_frontier_document() -> None:
    """A rank column is a composite score with the arithmetic hidden in the
    sort (I1)."""
    objectives, clusters, metrics = _sweep_fixture()
    result = rank_configs(
        ["fast", "safe", "bad"], metrics, clusters, objectives, constraints=()
    )
    payload = frontier_payload(result, {"fast": "t=1.0", "safe": "t=0.0"})

    def walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                assert key not in FORBIDDEN_FIELDS, f"{path}.{key}"
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for item in node:
                walk(item, path)

    walk(payload)


def test_the_frontier_document_carries_every_comparison_with_its_threshold() -> None:
    """D43: a reader who disagrees with a verdict must be able to see it."""
    objectives, clusters, metrics = _sweep_fixture()
    result = rank_configs(
        ["fast", "safe", "bad"], metrics, clusters, objectives, constraints=()
    )
    payload = frontier_payload(result)
    assert len(payload["comparisons"]) == 6
    for comparison in payload["comparisons"]:
        assert "p_pair" in comparison
        assert "holm_threshold" in comparison
        assert comparison["per_objective"]


def test_the_published_weighting_travels_with_the_objective() -> None:
    """I1: within-family weights are published, not implied."""
    objectives, clusters, metrics = _sweep_fixture()
    result = rank_configs(
        ["fast", "safe"], metrics, clusters, objectives, constraints=()
    )
    payload = frontier_payload(result)
    for objective in payload["objectives"]:
        assert objective["weighting"], objective["id"]


def test_a_relative_margin_is_reported_in_the_metrics_own_units() -> None:
    """Printing 0.10 beside a latency difference in milliseconds is not true."""
    objectives, clusters, metrics = _sweep_fixture()
    result = rank_configs(
        ["fast", "safe"], metrics, clusters, objectives, constraints=()
    )
    payload = frontier_payload(result)
    latency = next(
        c
        for comparison in payload["comparisons"]
        for c in comparison["per_objective"]
        if c["objective"] == "latency_p95_ms"
    )
    # 0.50: measured, not chosen. A smaller margin cannot establish
    # non-inferiority on latency, and domination needs it on every objective.
    assert latency["min_effect"] == pytest.approx(0.50)
    assert latency["applied_margin"] > 1.0


def test_a_coverage_gap_blocks_the_domination_it_would_otherwise_license() -> None:
    """§14.5, checked through the ranker rather than only on the matrix.

    Without the block, ``full`` dominates ``gappy`` on a comparison whose
    matched blocks do not line up.
    """
    objectives, clusters, metrics = _sweep_fixture()
    clusters = dict(clusters)
    metrics = dict(metrics)
    clusters["gappy"] = clusters.pop("bad")
    metrics["gappy"] = metrics.pop("bad")

    units = [f"u{i}" for i in range(12)]
    observations = {
        "fast": [_obs("fast", u, "security", Verdict.PASS) for u in units],
        "safe": [_obs("safe", u, "security", Verdict.PASS) for u in units],
        "gappy": [
            _obs("gappy", u, "security", Verdict.PASS if i < 8 else Verdict.UNSCORABLE)
            for i, u in enumerate(units)
        ],
    }

    without = rank_configs(
        ["fast", "safe", "gappy"], metrics, clusters, objectives, constraints=()
    )
    assert "gappy" in without.dominated, "the fixture no longer exercises the block"

    with_coverage = rank_configs(
        ["fast", "safe", "gappy"],
        metrics,
        clusters,
        objectives,
        constraints=(),
        coverage=coverage_matrix(observations, families=("security",)),
    )
    assert "gappy" not in with_coverage.dominated
    assert with_coverage.blocked_pairs
