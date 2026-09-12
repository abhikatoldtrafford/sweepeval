"""Sweep planning and the pre-flight budget (spec §12.1-§12.3, D22)."""

from __future__ import annotations

import pytest

from sweepeval.capabilities.detect import (
    Capability,
    CapabilityReport,
    CapabilityResult,
    Support,
)
from sweepeval.capabilities.sampling import SamplingVerdict, Verdict
from sweepeval.corpus.loader import load_corpus
from sweepeval.execute.budget import BudgetCap, estimate_run, render_estimate
from sweepeval.execute.planner import (
    CAP_BY_PROFILE,
    MODEL_PROBE_BOUND,
    SHRINK_LADDER,
    filter_model_ids,
    plan_sweep,
)


def _caps(system_prompt: bool = True) -> CapabilityReport:
    report = CapabilityReport()
    report.results[Capability.SYSTEM_PROMPT] = CapabilityResult(
        Capability.SYSTEM_PROMPT,
        Support.SUPPORTED if system_prompt else Support.UNSUPPORTED,
        "behavioural probe", "high",
    )
    return report


def _sampling(**verdicts: Verdict) -> dict[str, SamplingVerdict]:
    return {
        name: SamplingVerdict(parameter=name, verdict=v, tier=2, reason="test")
        for name, v in verdicts.items()
    }


# --- the cap is an integer (D22) ------------------------------------------


def test_the_cap_is_a_single_integer_per_profile() -> None:
    """"8-12" cannot be a deterministic shrink target."""
    assert CAP_BY_PROFILE == {"quick": 6, "standard": 12, "deep": 12}
    for value in CAP_BY_PROFILE.values():
        assert isinstance(value, int)


def test_the_sweep_never_exceeds_its_cap() -> None:
    plan = plan_sweep(
        _caps(), [f"model-{i}" for i in range(8)],
        profile="standard", sampling=_sampling(temperature=Verdict.EFFECTIVE),
    )
    assert len(plan.configs) <= CAP_BY_PROFILE["standard"]


def test_quick_caps_lower_than_standard() -> None:
    args = dict(sampling=_sampling(temperature=Verdict.EFFECTIVE))
    quick = plan_sweep(_caps(), ["m1", "m2", "m3"], profile="quick", **args)  # type: ignore[arg-type]
    standard = plan_sweep(_caps(), ["m1", "m2", "m3"], profile="standard", **args)  # type: ignore[arg-type]
    assert len(quick.configs) <= len(standard.configs)


# --- model filtering and ordering (§12.2) ---------------------------------


def test_non_chat_models_are_dropped_with_a_reason() -> None:
    """A gateway lists embeddings and TTS; sweeping them as chat configs is a
    guaranteed terminal-error storm."""
    kept, dropped = filter_model_ids(
        ["gpt-chat", "text-embedding-3", "whisper-1", "tts-1", "claude-chat"]
    )
    assert kept == ["claude-chat", "gpt-chat"]
    assert {m for m, _ in dropped} == {"text-embedding-3", "whisper-1", "tts-1"}
    for _, reason in dropped:
        assert reason


def test_models_are_sorted_before_the_bound_applies() -> None:
    """Server ordering is not stable.

    Bounding an unsorted list makes "the same target always yields the same
    sweep" false, which is the property D22 rests on.
    """
    forward = filter_model_ids([f"m{i:02d}" for i in range(30)])[0]
    reversed_input = filter_model_ids([f"m{i:02d}" for i in reversed(range(30))])[0]
    assert forward == reversed_input


def test_the_probe_loop_is_bounded() -> None:
    """An unbounded probe loop is itself an unbudgeted spend path."""
    kept, dropped = filter_model_ids([f"m{i:03d}" for i in range(100)])
    assert len(kept) == MODEL_PROBE_BOUND
    assert any(r == "beyond_probe_bound" for _, r in dropped)


# --- axis selection --------------------------------------------------------


def test_an_unsupported_system_prompt_is_not_an_axis() -> None:
    """A target that drops the role would give four identical configs."""
    plan = plan_sweep(_caps(system_prompt=False), ["m1"])
    assert "system_prompt" not in plan.axes
    assert any(a == "system_prompt" for a, _ in plan.rejected_axes)


def test_an_inert_temperature_is_not_swept() -> None:
    """§9.1: sweeping a parameter proved inert is theatre."""
    plan = plan_sweep(_caps(), ["m1"], sampling=_sampling(temperature=Verdict.INERT))
    assert "temperature" not in plan.axes
    assert any("INERT" in r for a, r in plan.rejected_axes if a == "temperature")


def test_an_inconclusive_temperature_is_swept_anyway() -> None:
    """§9.1's asymmetry: an inert axis costs money, a missed one costs the
    experiment."""
    plan = plan_sweep(
        _caps(), ["m1"], sampling=_sampling(temperature=Verdict.INCONCLUSIVE)
    )
    assert "temperature" in plan.axes


def test_top_p_is_not_swept_alongside_temperature() -> None:
    """Two knobs that move the same thing double the config count."""
    plan = plan_sweep(
        _caps(), ["m1"],
        sampling=_sampling(temperature=Verdict.EFFECTIVE, top_p=Verdict.EFFECTIVE),
    )
    assert "temperature" in plan.axes
    assert "top_p" not in plan.axes


def test_top_p_is_swept_when_temperature_is_inert() -> None:
    plan = plan_sweep(
        _caps(), ["m1"], profile="standard",
        sampling=_sampling(temperature=Verdict.INERT, top_p=Verdict.EFFECTIVE),
    )
    assert "top_p" in plan.axes


def test_top_p_is_the_ladders_first_casualty_when_over_cap() -> None:
    """D22's ladder drops top_p first, so at `quick` (cap 6) a 4-variant
    system axis plus 3 top_p values loses top_p rather than shrinking both."""
    plan = plan_sweep(
        _caps(), ["m1"], profile="quick",
        sampling=_sampling(temperature=Verdict.INERT, top_p=Verdict.EFFECTIVE),
    )
    assert "top_p" not in plan.axes
    assert any(step.startswith("drop top_p") for step in plan.shrink_steps)


def test_a_single_model_is_not_an_axis() -> None:
    plan = plan_sweep(_caps(), ["only-one"])
    assert "model" not in plan.axes


# --- the shrink ladder (D22) ----------------------------------------------


def test_the_ladder_is_fixed_and_ordered() -> None:
    assert SHRINK_LADDER == (
        "drop top_p",
        "drop temperature=0.7",
        "drop system_prompt=terse_permissive",
        "cap models",
        "truncate to the cap",
    )


def test_the_cap_binds_even_when_the_axis_steps_cannot_reach_it() -> None:
    """The pre-flight estimate is computed from the cap (§12.3), so a plan
    that exceeded it would spend more than the user consented to."""
    plan = plan_sweep(
        _caps(), ["m1", "m2"], profile="quick", cap=4,
        sampling=_sampling(temperature=Verdict.EFFECTIVE),
    )
    assert len(plan.configs) == 4
    assert any(s.startswith("truncate to the cap") for s in plan.shrink_steps)


def test_truncation_is_the_last_resort_not_the_first() -> None:
    """Dropping an axis value is preferable to trimming the design: the
    axis-level steps are disclosed and balanced, truncation is neither."""
    plan = plan_sweep(
        _caps(), ["m1", "m2"], profile="standard",
        sampling=_sampling(temperature=Verdict.EFFECTIVE),
    )
    assert len(plan.configs) <= 12
    assert not any(s.startswith("truncate") for s in plan.shrink_steps)


def test_every_shrink_step_taken_is_recorded() -> None:
    """A user whose model axis vanished must be able to see why."""
    plan = plan_sweep(
        _caps(), [f"m{i}" for i in range(6)],
        profile="standard", sampling=_sampling(temperature=Verdict.EFFECTIVE),
    )
    assert plan.shrink_steps
    for step in plan.shrink_steps:
        assert "configs" in step


def test_planning_is_deterministic() -> None:
    """D22: the same target always yields the same sweep."""
    args = dict(
        profile="standard", sampling=_sampling(temperature=Verdict.EFFECTIVE)
    )
    first = plan_sweep(_caps(), ["b", "a", "c"], **args)  # type: ignore[arg-type]
    second = plan_sweep(_caps(), ["c", "b", "a"], **args)  # type: ignore[arg-type]
    assert [c.label() for c in first.configs] == [c.label() for c in second.configs]


def test_no_axes_means_a_single_configuration() -> None:
    """D15: a sweep with nothing to sweep is an evaluation."""
    plan = plan_sweep(_caps(system_prompt=False), ["m1"])
    assert plan.is_single_config
    assert len(plan.configs) == 1
    assert plan.rejected_axes


def test_config_ids_are_stable_and_labelled() -> None:
    plan = plan_sweep(_caps(), ["m1"], sampling=_sampling(temperature=Verdict.EFFECTIVE))
    assert len({c.config_id for c in plan.configs}) == len(plan.configs)
    for config in plan.configs:
        assert "sys=" in config.label()


def test_the_system_prompt_text_is_attached_not_just_its_name() -> None:
    plan = plan_sweep(_caps(), ["m1"])
    variants = {c.system_prompt_variant: c.system_prompt for c in plan.configs}
    assert variants["none"] is None
    assert variants["verbose_strict_with_guardrails"]
    assert "never reveal" in variants["verbose_strict_with_guardrails"]


# --- the pre-flight budget (§12.3, I9) ------------------------------------


def test_the_estimate_covers_every_phase_not_just_scoring() -> None:
    """I9. Discovery and capability detection spend before the planner runs,
    and an earlier revision gated after they had."""
    estimate = estimate_run(
        load_corpus("quick"), configs=6, runs=3, profile="quick"
    )
    phases = {p.phase for p in estimate.phases}
    assert "discovery" in phases
    assert "capabilities" in phases
    assert any(p.startswith("scoring") for p in phases)


def test_the_scoring_line_sums_calls_not_units() -> None:
    """`configs x units x runs` is wrong for every multi-turn unit."""
    corpus = load_corpus("standard")
    estimate = estimate_run(corpus, configs=12, runs=3, profile="standard")
    scoring = next(p for p in estimate.phases if p.phase.startswith("scoring"))
    assert scoring.requests == corpus.calls_per_run * 3 * 12
    assert scoring.requests > corpus.unit_count * 3 * 12


def test_hard_fail_confirmations_are_budgeted() -> None:
    """§11.2: three re-runs per hard-fail-capable unit, usually unspent."""
    corpus = load_corpus("quick")
    without = estimate_run(corpus, configs=1, runs=3, profile="quick")
    with_confirm = estimate_run(
        corpus, configs=1, runs=3, profile="quick", hard_fail_units=3
    )
    assert with_confirm.total_requests > without.total_requests


def test_the_rendered_block_leads_with_profile_and_totals() -> None:
    estimate = estimate_run(load_corpus("quick"), configs=6, runs=3, profile="quick")
    text = render_estimate(estimate)
    assert text.splitlines()[0].strip().startswith("profile")
    assert "total" in text
    assert "wall-clock" in text


def test_quick_is_marked_not_gate_eligible_in_the_estimate() -> None:
    text = render_estimate(
        estimate_run(load_corpus("quick"), configs=6, runs=3, profile="quick")
    )
    assert "not gate-eligible" in text


def test_no_pricing_says_so_rather_than_implying_zero_cost() -> None:
    text = render_estimate(
        estimate_run(load_corpus("quick"), configs=1, runs=3, profile="quick")
    )
    assert "no pricing supplied" in text
    assert "tokens_out_per_probe" in text


def test_a_request_cap_binds() -> None:
    estimate = estimate_run(load_corpus("standard"), configs=12, runs=3, profile="standard")
    assert BudgetCap(value=100, unit="requests").exceeded_by(estimate)
    assert not BudgetCap(value=10_000_000, unit="requests").exceeded_by(estimate)


def test_an_uncapped_budget_never_binds() -> None:
    estimate = estimate_run(load_corpus("quick"), configs=6, runs=3, profile="quick")
    assert not BudgetCap().exceeded_by(estimate)


def test_a_dollar_cap_cannot_bind_without_pricing() -> None:
    """Treating "no pricing" as "no cost" would silently disable the cap."""
    estimate = estimate_run(load_corpus("quick"), configs=6, runs=3, profile="quick")
    assert not BudgetCap(value=0.01, unit="dollars").exceeded_by(estimate)


def test_wall_clock_scales_with_concurrency() -> None:
    corpus = load_corpus("quick")
    serial = estimate_run(corpus, configs=6, runs=3, profile="quick", concurrency=1)
    parallel = estimate_run(corpus, configs=6, runs=3, profile="quick", concurrency=4)
    assert parallel.wall_clock_minutes < serial.wall_clock_minutes


@pytest.mark.parametrize("profile", ["quick", "standard"])
def test_the_estimate_is_a_real_number_for_every_profile(profile: str) -> None:
    corpus = load_corpus(profile)  # type: ignore[arg-type]
    estimate = estimate_run(
        corpus, configs=CAP_BY_PROFILE[profile], runs=3, profile=profile
    )
    assert estimate.total_requests > 0
    assert estimate.total_tokens > 0
    assert estimate.wall_clock_minutes > 0
