"""Determinism and context scorers (spec §11.4, §11.5)."""

from __future__ import annotations

import pytest

from sweepeval.schema.observation import Verdict
from sweepeval.schema.unit import ScoringContract, Turn, Unit
from sweepeval.scorers import RunEvidence, ScoreContext, registry
from sweepeval.scorers.context import depth_at_floor, retention_auc, retention_curve


def _ctx() -> ScoreContext:
    return ScoreContext(
        run_id="r1", config_id="c1", run_idx=0, text="", ts="2026-09-12T00:00:00Z"
    )


def _det_unit(template_id: str = "det.base.apology.v1") -> Unit:
    return Unit.make(
        template_id=template_id, family="determinism",
        turns=[Turn(role="user", text="draft something")],
        scoring=[ScoringContract(kind="equivalence")], profiles={"standard"},
    )


def _ctx_unit(depth: int, expect: str = "48812") -> Unit:
    return Unit.make(
        template_id=f"ctx.order.d{depth}.v1", family="context",
        turns=[Turn(role="user", text="q")] * max(1, depth),
        scoring=[ScoringContract(kind="fact_recall", expect=expect)],
        profiles={"standard"}, params={"depth": depth}, depth=depth,
    )


def _finalize(evidence: list[RunEvidence]) -> list:
    return registry().get("determinism").finalize(evidence, _ctx())  # type: ignore[attr-defined]


# --- determinism: repeatability -------------------------------------------


def test_identical_runs_are_perfectly_repeatable() -> None:
    obs = _finalize([RunEvidence(_det_unit(), ("same", "same", "same"))])
    value = next(o for o in obs if o.metric == "config_repeatability")
    assert value.value == 1.0


def test_all_different_runs_are_not_repeatable() -> None:
    obs = _finalize([RunEvidence(_det_unit(), ("alpha", "beta", "gamma"))])
    value = next(o for o in obs if o.metric == "config_repeatability")
    assert value.value == 0.0


def test_repeatability_is_a_fraction_of_pairs() -> None:
    obs = _finalize([RunEvidence(_det_unit(), ("same", "same", "other"))])
    value = next(o for o in obs if o.metric == "config_repeatability")
    assert value.value == pytest.approx(1 / 3)


def test_a_timestamp_does_not_count_as_nondeterminism() -> None:
    """§9.1's normaliser applies here too.

    A target that stamps its replies would otherwise score 0% repeatable for a
    reason that has nothing to do with sampling.
    """
    obs = _finalize([
        RunEvidence(_det_unit(), (
            "Done at 2026-09-12T10:00:00Z",
            "Done at 2026-09-12T11:30:00Z",
        ))
    ])
    value = next(o for o in obs if o.metric == "config_repeatability")
    assert value.value == 1.0


# --- determinism: naming (§11.4, §14.2) -----------------------------------


def test_both_determinism_metrics_are_emitted_when_temperature_is_not_swept() -> None:
    """They coincide, but the distinction has to survive an axis appearing."""
    obs = _finalize([RunEvidence(_det_unit(), ("a", "a"))])
    metrics = {o.metric for o in obs}
    assert "config_repeatability" in metrics
    assert "target_determinism_at_temp0" in metrics


def test_the_coincidence_is_stated_in_the_reason() -> None:
    obs = _finalize([RunEvidence(_det_unit(), ("a", "a"))])
    target = next(o for o in obs if o.metric == "target_determinism_at_temp0")
    assert "not a swept axis" in (target.reason or "")


def test_the_temp0_metric_is_withheld_when_temperature_is_swept() -> None:
    """Reporting a temp=0 number on a temp=1.0 row is the misleading case
    §11.4 exists to prevent; it needs its own measurement, not this one."""
    from sweepeval.scorers.determinism import DeterminismScorer

    scorer = DeterminismScorer(temperature_is_swept=True)
    obs = scorer.finalize([RunEvidence(_det_unit(), ("a", "a"))], _ctx())
    assert "target_determinism_at_temp0" not in {o.metric for o in obs}
    assert "config_repeatability" in {o.metric for o in obs}


# --- determinism: exclusions and stability --------------------------------


def test_one_scorable_run_cannot_measure_repeatability() -> None:
    obs = _finalize([RunEvidence(_det_unit(), ("only", "", ""), unscorable=(1, 2))])
    for observation in obs:
        assert observation.verdict is Verdict.UNSCORABLE
        assert "fewer than two" in (observation.reason or "")


def test_semantic_stability_sits_between_exact_and_nothing() -> None:
    obs = _finalize([
        RunEvidence(_det_unit(), ("the parcel arrived late", "the parcel arrived early"))
    ])
    semantic = next(o for o in obs if o.metric == "semantic_stability")
    exact = next(o for o in obs if o.metric == "config_repeatability")
    assert exact.value == 0.0
    assert semantic.value is not None
    assert 0.0 < semantic.value < 1.0, "paraphrases share most tokens"


# --- determinism: invariance ----------------------------------------------


def test_paraphrases_with_the_same_answer_are_invariant() -> None:
    group = [
        RunEvidence(_det_unit("det.inv.refund.a.v1"), ("thirty days",)),
        RunEvidence(_det_unit("det.inv.refund.b.v1"), ("thirty days",)),
        RunEvidence(_det_unit("det.inv.refund.c.v1"), ("thirty days",)),
    ]
    obs = _finalize(group)
    invariance = next(o for o in obs if o.metric == "invariance")
    assert invariance.value == 1.0


def test_paraphrases_with_different_answers_are_not_invariant() -> None:
    group = [
        RunEvidence(_det_unit("det.inv.refund.a.v1"), ("thirty days",)),
        RunEvidence(_det_unit("det.inv.refund.b.v1"), ("no returns accepted",)),
    ]
    obs = _finalize(group)
    invariance = next(o for o in obs if o.metric == "invariance")
    assert invariance.value is not None and invariance.value < 0.5


def test_an_invariance_group_produces_one_observation_not_one_per_member() -> None:
    group = [
        RunEvidence(_det_unit(f"det.inv.refund.{v}.v1"), ("same",))
        for v in ("a", "b", "c")
    ]
    assert sum(1 for o in _finalize(group) if o.metric == "invariance") == 1


def test_base_prompts_and_invariance_groups_are_separated() -> None:
    """§10.2: base prompts are the CLUSTERS; groups belong to another
    sub-scorer, and mixing them would inflate the cluster count."""
    evidence = [
        RunEvidence(_det_unit("det.base.apology.v1"), ("x", "x")),
        RunEvidence(_det_unit("det.inv.refund.a.v1"), ("y",)),
        RunEvidence(_det_unit("det.inv.refund.b.v1"), ("y",)),
    ]
    obs = _finalize(evidence)
    assert sum(1 for o in obs if o.metric == "config_repeatability") == 1
    assert sum(1 for o in obs if o.metric == "invariance") == 1


# --- context: recall -------------------------------------------------------


def test_a_recalled_fact_passes() -> None:
    scorer = registry().get("context")
    context = ScoreContext(
        run_id="r", config_id="c", run_idx=0,
        text="Your order number is 48812.", ts="t",
    )
    obs = scorer.score(_ctx_unit(8), [], context)
    assert obs[0].verdict is Verdict.PASS
    assert obs[0].depth == 8


def test_a_lost_fact_fails() -> None:
    scorer = registry().get("context")
    context = ScoreContext(
        run_id="r", config_id="c", run_idx=0,
        text="I don't recall that from earlier.", ts="t",
    )
    assert scorer.score(_ctx_unit(15), [], context)[0].verdict is Verdict.FAIL


def test_no_extracted_text_is_unscorable_not_a_failure() -> None:
    """I5: a transport problem is not evidence the target forgot."""
    scorer = registry().get("context")
    context = ScoreContext(run_id="r", config_id="c", run_idx=0, text="", ts="t")
    obs = scorer.score(_ctx_unit(3), [], context)
    assert obs[0].verdict is Verdict.UNSCORABLE


# --- context: the decay curve and AUC (§11.5) -----------------------------


def _recall_obs(depth: int, passed: bool):
    scorer = registry().get("context")
    context = ScoreContext(
        run_id="r", config_id="c", run_idx=0,
        text="order 48812" if passed else "no idea", ts="t",
    )
    return scorer.score(_ctx_unit(depth), [], context)[0]


def test_the_curve_averages_recall_at_each_depth() -> None:
    observations = [
        _recall_obs(3, True), _recall_obs(3, True),
        _recall_obs(8, True), _recall_obs(8, False),
        _recall_obs(15, False), _recall_obs(15, False),
    ]
    assert retention_curve(observations) == {3: 1.0, 8: 0.5, 15: 0.0}


def test_the_auc_is_the_normalised_trapezoid() -> None:
    auc, weights = retention_auc({3: 1.0, 8: 0.5, 15: 0.0})
    assert 0.0 < auc < 1.0
    assert sum(weights.values()) == pytest.approx(1.0)


def test_depth_spacing_sets_the_weights() -> None:
    """§11.5: on the 3/8/15 ladder the middle depth carries roughly half the
    area purely from geometry, which is why profile is a hard key."""
    _, weights = retention_auc({3: 1.0, 8: 1.0, 15: 1.0})
    assert weights[8] > weights[3]
    assert weights[8] > weights[15]
    assert weights[8] == pytest.approx(0.5)


def test_a_deeper_ladder_changes_the_weights() -> None:
    """Adding depth 30 under `deep` redefines the quantity."""
    _, standard = retention_auc({3: 1.0, 8: 1.0, 15: 1.0})
    _, deep = retention_auc({3: 1.0, 8: 1.0, 15: 1.0, 30: 1.0})
    assert standard != deep


def test_a_single_depth_has_no_area_and_says_so() -> None:
    """§10.2: at `quick` the AUC degrades to that depth's recall."""
    auc, weights = retention_auc({3: 0.8})
    assert auc == 0.8
    assert weights == {3: 1.0}


def test_a_perfect_curve_scores_one() -> None:
    auc, _ = retention_auc({3: 1.0, 8: 1.0, 15: 1.0})
    assert auc == pytest.approx(1.0)


def test_a_total_failure_scores_zero() -> None:
    auc, _ = retention_auc({3: 0.0, 8: 0.0, 15: 0.0})
    assert auc == pytest.approx(0.0)


def test_depth_at_floor_reports_where_retention_breaks() -> None:
    assert depth_at_floor({3: 1.0, 8: 0.8, 15: 0.2}) == 15
    assert depth_at_floor({3: 1.0, 8: 1.0, 15: 1.0}) is None


def test_unscorable_trials_are_excluded_from_the_curve() -> None:
    """§11.8: a refused trial leaves the metric rather than counting as a loss."""
    scorer = registry().get("context")
    good = _recall_obs(3, True)
    empty = scorer.score(
        _ctx_unit(8), [], ScoreContext(run_id="r", config_id="c", run_idx=0, text="", ts="t")
    )[0]
    assert retention_curve([good, empty]) == {3: 1.0}
