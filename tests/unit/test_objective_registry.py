"""The objective registry (spec §14.1, §11.4, §14.2). I1 and I10 load-bearing."""

from __future__ import annotations

import pytest

from sweepeval.schema.objective import REGISTRY, Objective, ObjectiveRegistry


def _objective(**over: object) -> Objective:
    base: dict[str, object] = dict(
        id="my_metric",
        display_label="My Metric",
        direction="maximize",
        family="custom",
        cluster_key="probe",
        min_effect=0.01,
        min_effect_kind="absolute",
        default=False,
    )
    base.update(over)
    return Objective(**base)  # type: ignore[arg-type]


# --- the shipped set ------------------------------------------------------


def test_the_six_default_objectives_are_registered() -> None:
    assert {o.id for o in REGISTRY.defaults()} == {
        "security_pass_rate",
        "guardrail_pass_rate",
        "target_determinism_at_temp0",
        "context_retention_auc",
        "latency_p95_ms",
        "cost_per_probe",
    }


def test_config_repeatability_is_registered_but_not_a_default() -> None:
    objective = REGISTRY.get("config_repeatability")
    assert objective.default is False
    assert objective.id not in {d.id for d in REGISTRY.defaults()}


def test_directions_match_the_spec_table() -> None:
    directions = {o.id: o.direction for o in REGISTRY.defaults()}
    assert directions["security_pass_rate"] == "maximize"
    assert directions["guardrail_pass_rate"] == "maximize"
    assert directions["target_determinism_at_temp0"] == "maximize"
    assert directions["context_retention_auc"] == "maximize"
    assert directions["latency_p95_ms"] == "minimize"
    assert directions["cost_per_probe"] == "minimize"


def test_min_effects_match_the_spec_table() -> None:
    effects = {o.id: (o.min_effect, o.min_effect_kind) for o in REGISTRY.defaults()}
    assert effects["security_pass_rate"] == (0.02, "absolute")
    assert effects["guardrail_pass_rate"] == (0.02, "absolute")
    assert effects["latency_p95_ms"] == (0.10, "relative")
    assert effects["cost_per_probe"] == (0.10, "relative")


def test_cluster_keys_match_the_spec_table() -> None:
    keys = {o.id: o.cluster_key for o in REGISTRY.defaults()}
    assert keys["security_pass_rate"] == "security_probe"
    assert keys["guardrail_pass_rate"] == "guardrail_probe"
    assert keys["target_determinism_at_temp0"] == "determinism_base_prompt"
    assert keys["context_retention_auc"] == "conversation"
    assert keys["latency_p95_ms"] == "probe"
    assert keys["cost_per_probe"] == "probe"


def test_every_objective_declares_a_min_effect() -> None:
    """§13.6: significance alone must never decide domination or a build."""
    for objective in REGISTRY.all():
        assert objective.min_effect > 0, objective.id


def test_every_objective_declares_its_within_family_weighting() -> None:
    """I1: within-family aggregation is legitimate but must be stated."""
    for objective in REGISTRY.all():
        assert objective.weighting, objective.id


# --- the determinism pair -------------------------------------------------


def test_the_determinism_objective_names_what_it_actually_measures() -> None:
    objective = REGISTRY.get("target_determinism_at_temp0")
    assert objective.cluster_key == "determinism_base_prompt"
    assert "model, system_prompt" in objective.note


def test_the_two_determinism_metrics_are_distinct_and_both_present() -> None:
    target = REGISTRY.get("target_determinism_at_temp0")
    config = REGISTRY.get("config_repeatability")
    assert target.id != config.id
    assert target.default
    assert not config.default
    assert target.family == config.family == "determinism"


def test_config_repeatability_explains_why_it_is_not_default() -> None:
    assert "temperature axis" in REGISTRY.get("config_repeatability").note


# --- plugin surface (I10) -------------------------------------------------


def test_a_plugin_objective_registers_without_touching_rank() -> None:
    registry = ObjectiveRegistry()
    registry.register(_objective())
    assert registry.get("my_metric").family == "custom"
    assert "my_metric" not in {o.id for o in registry.defaults()}


def test_a_plugin_objective_can_be_a_default() -> None:
    registry = ObjectiveRegistry()
    registry.register(_objective(default=True))
    assert [o.id for o in registry.defaults()] == ["my_metric"]


def test_duplicate_registration_is_rejected() -> None:
    """Silently replacing an objective would change what a frontier axis means."""
    registry = ObjectiveRegistry()
    registry.register(_objective())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_objective())


def test_unknown_objective_error_lists_the_known_ones() -> None:
    with pytest.raises(KeyError) as excinfo:
        REGISTRY.get("nope")
    assert "security_pass_rate" in str(excinfo.value)


def test_registry_is_ordered_deterministically() -> None:
    """Report and frontier column order must not depend on registration order."""
    ids = [o.id for o in REGISTRY.defaults()]
    assert ids == sorted(ids)
    all_ids = [o.id for o in REGISTRY.all()]
    assert all_ids == sorted(all_ids)


def test_objectives_are_frozen() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        REGISTRY.get("security_pass_rate").min_effect = 0.5  # type: ignore[misc]
