"""The determinism confound (spec §14.2, §11.4).

``target_determinism_at_temp0`` measured at each config's own settings is ~1.0
for every temp=0 config and ~0 for every temp=1.0 config *by construction*.
That is a manufactured win on a metric that merely restates the row's own
label, and it makes every temp=0 config non-dominated for free.

The fix relocates the structure rather than removing it: one value per (model,
system_prompt) pair, exactly tied within each temperature triple. A tie does
not block domination the way a manufactured win does.
"""

from __future__ import annotations

from sweepeval.corpus.loader import load_corpus
from sweepeval.execute.aggregation import ConfigAggregate
from sweepeval.execute.planner import ConfigSpec
from sweepeval.execute.sweep import (
    ConfigResult,
    SweepPlan,
    SweepResult,
    SweepStatus,
    _determinism_sharing,
    _share_determinism,
)
from sweepeval.schema.metric import CIMethod, Estimand, MetricValue

METRIC = "target_determinism_at_temp0"


def _mv(point: float) -> MetricValue:
    return MetricValue(
        point=point, lo=max(0.0, point - 0.05), hi=min(1.0, point + 0.05),
        method=CIMethod.cluster_bootstrap, n_clusters=12, alpha=0.05,
        estimand=Estimand.generalization,
    )


def _spec(config_id: str, model: str, temperature: float, variant: str) -> ConfigSpec:
    return ConfigSpec(
        config_id=config_id,
        params={"model": model, "temperature": temperature},
        system_prompt=None,
        system_prompt_variant=variant,
    )


def _result(specs, determinism, repeatability):
    rows = []
    for spec in specs:
        aggregate = ConfigAggregate(config_id=spec.config_id)
        aggregate.metrics[METRIC] = _mv(determinism[spec.config_id])
        aggregate.metrics["config_repeatability"] = _mv(
            repeatability[spec.config_id]
        )
        aggregate.clusters[METRIC] = {"p1": determinism[spec.config_id]}
        rows.append(ConfigResult(config=spec, aggregate=aggregate))

    return SweepResult(
        run_id="r",
        profile="quick",
        runs=3,
        status=SweepStatus.COMPLETE,
        corpus=load_corpus("quick"),
        plan=SweepPlan(configs=tuple(specs)),
        configs=rows,
    )


def _triple():
    return [
        _spec("cfg-00", "m1", 0.0, "none"),
        _spec("cfg-01", "m1", 0.7, "none"),
        _spec("cfg-02", "m1", 1.0, "none"),
    ]


def test_a_temperature_sibling_inherits_the_temp0_figure() -> None:
    specs = _triple()
    result = _result(
        specs,
        determinism={"cfg-00": 1.0, "cfg-01": 0.2, "cfg-02": 0.0},
        repeatability={"cfg-00": 1.0, "cfg-01": 0.2, "cfg-02": 0.0},
    )
    _share_determinism(result, _determinism_sharing(result.plan))

    values = {r.config_id: r.metrics[METRIC].point for r in result.configs}
    assert values == {"cfg-00": 1.0, "cfg-01": 1.0, "cfg-02": 1.0}


def test_the_shared_clusters_travel_with_the_shared_point() -> None:
    """The paired test resamples them; a borrowed point beside the row's own
    variance compares one config's number against another's noise."""
    specs = _triple()
    result = _result(
        specs,
        determinism={"cfg-00": 1.0, "cfg-01": 0.2, "cfg-02": 0.0},
        repeatability={"cfg-00": 1.0, "cfg-01": 0.2, "cfg-02": 0.0},
    )
    _share_determinism(result, _determinism_sharing(result.plan))
    for row in result.configs:
        assert row.clusters[METRIC] == {"p1": 1.0}


def test_repeatability_at_own_settings_is_left_alone() -> None:
    """It is the production-truth number, reported beside the shared one."""
    specs = _triple()
    result = _result(
        specs,
        determinism={"cfg-00": 1.0, "cfg-01": 0.2, "cfg-02": 0.0},
        repeatability={"cfg-00": 1.0, "cfg-01": 0.2, "cfg-02": 0.0},
    )
    _share_determinism(result, _determinism_sharing(result.plan))
    own = {r.config_id: r.metrics["config_repeatability"].point for r in result.configs}
    assert own == {"cfg-00": 1.0, "cfg-01": 0.2, "cfg-02": 0.0}


def test_a_different_system_prompt_gets_its_own_owner() -> None:
    """The objective is a property of the (model, system_prompt) pair, so two
    pairs must not share one number."""
    specs = [
        _spec("cfg-00", "m1", 0.0, "none"),
        _spec("cfg-01", "m1", 1.0, "none"),
        _spec("cfg-02", "m1", 0.0, "strict"),
        _spec("cfg-03", "m1", 1.0, "strict"),
    ]
    result = _result(
        specs,
        determinism={"cfg-00": 1.0, "cfg-01": 0.0, "cfg-02": 0.6, "cfg-03": 0.0},
        repeatability=dict.fromkeys(("cfg-00", "cfg-01", "cfg-02", "cfg-03"), 0.0),
    )
    _share_determinism(result, _determinism_sharing(result.plan))
    values = {r.config_id: r.metrics[METRIC].point for r in result.configs}
    assert values == {"cfg-00": 1.0, "cfg-01": 1.0, "cfg-02": 0.6, "cfg-03": 0.6}


def test_a_row_whose_owner_never_ran_keeps_its_own_number_and_says_so() -> None:
    """A shared figure attributed to a config that was never executed is worse
    than an honest self-measurement."""
    specs = _triple()
    result = _result(
        specs,
        determinism={"cfg-00": 1.0, "cfg-01": 0.2, "cfg-02": 0.0},
        repeatability=dict.fromkeys(("cfg-00", "cfg-01", "cfg-02"), 0.0),
    )
    # The budget cap stopped before the temp=0 owner.
    result.configs = [r for r in result.configs if r.config_id != "cfg-00"]

    _share_determinism(result, _determinism_sharing(result.plan))
    assert result.configs[0].metrics[METRIC].point == 0.2
    assert "own settings" in result.determinism_scope["cfg-01"]


def test_the_scope_map_names_where_each_number_came_from() -> None:
    specs = _triple()
    result = _result(
        specs,
        determinism={"cfg-00": 1.0, "cfg-01": 0.2, "cfg-02": 0.0},
        repeatability=dict.fromkeys(("cfg-00", "cfg-01", "cfg-02"), 0.0),
    )
    _share_determinism(result, _determinism_sharing(result.plan))
    assert result.determinism_scope == {
        "cfg-00": "cfg-00",
        "cfg-01": "cfg-00",
        "cfg-02": "cfg-00",
    }


def test_without_a_temperature_axis_every_row_owns_its_own() -> None:
    specs = [
        _spec("cfg-00", "m1", 0.0, "none"),
        _spec("cfg-01", "m2", 0.0, "none"),
    ]
    result = _result(
        specs,
        determinism={"cfg-00": 1.0, "cfg-01": 0.4},
        repeatability={"cfg-00": 1.0, "cfg-01": 0.4},
    )
    _share_determinism(result, _determinism_sharing(result.plan))
    values = {r.config_id: r.metrics[METRIC].point for r in result.configs}
    assert values == {"cfg-00": 1.0, "cfg-01": 0.4}
