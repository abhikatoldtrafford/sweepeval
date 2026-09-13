"""Every objective clears the bootstrap floor at every profile (Task 4.2b).

A corpus edit that drops a family below the floor must fail CI. Left to
runtime it would silently delete an objective from the frontier: the metric
gets ``NO_VALID_INTERVAL``, domination can no longer be asserted on it, and the
report is quietly one dimension smaller than the spec says it is.

This is the guard for the defect that shipped in rev 2 of the spec, where the
quick profile was sized at 16 units and *every* objective fell below the floor.
"""

from __future__ import annotations

import pytest

from sweepeval.corpus.loader import CLUSTER_KEY_BY_FAMILY, load_corpus
from sweepeval.corpus.template import PROFILES
from sweepeval.stats.resample import CLUSTER_FLOOR

MARGIN = 2
"""Clusters above the floor a family must keep in reserve.

§11.8 makes a determinism or context refusal UNSCORABLE with the trial
excluded, so a family sitting exactly on the floor loses its interval to a
single refusal.
"""


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("family", sorted(CLUSTER_KEY_BY_FAMILY))
def test_every_family_clears_the_cluster_floor_with_margin(
    profile: str, family: str
) -> None:
    corpus = load_corpus(profile)  # type: ignore[arg-type]
    count = corpus.cluster_count(family)
    assert count >= CLUSTER_FLOOR + MARGIN, (
        f"{profile}/{family}: {count} clusters, needs "
        f"{CLUSTER_FLOOR + MARGIN} (floor {CLUSTER_FLOOR} + margin {MARGIN}). "
        "Below the floor this objective silently leaves the frontier."
    )


@pytest.mark.parametrize("profile", PROFILES)
def test_operational_clusters_are_every_probe(profile: str) -> None:
    """Latency and cost resample over probes, so they inherit the whole corpus."""
    corpus = load_corpus(profile)  # type: ignore[arg-type]
    assert corpus.unit_count >= CLUSTER_FLOOR + MARGIN


# --- the shape §10.2 fixes -------------------------------------------------


def test_the_standard_profile_shape() -> None:
    corpus = load_corpus("standard")
    assert corpus.cluster_count("security") == 24
    assert corpus.cluster_count("guardrail") == 20
    assert corpus.cluster_count("determinism") == 12
    assert corpus.cluster_count("context") == 12


def test_the_quick_profile_shape() -> None:
    """The generated shape, not an estimate.

    §10.1 requires this table to be derived from the templates. Asserting it
    here is what stops the spec's figures and the corpus drifting apart, which
    is how rev 1 came to under-count every multi-turn unit in the budget the
    user consents to.
    """
    corpus = load_corpus("quick")
    assert corpus.unit_count == 40
    assert corpus.calls_per_run == 60


def test_the_standard_profile_call_count() -> None:
    corpus = load_corpus("standard")
    assert corpus.unit_count == 80
    assert corpus.calls_per_run == 188


def test_the_budget_estimate_sums_calls_not_units() -> None:
    """§12.3: `configs x units x runs` is wrong for every multi-turn unit."""
    corpus = load_corpus("standard")
    estimate = corpus.estimate(configs=12, runs=3)
    assert estimate["total_calls"] == 188 * 3 * 12
    assert estimate["total_calls"] > corpus.unit_count * 3 * 12


def test_quick_is_cheaper_than_standard() -> None:
    quick = load_corpus("quick")
    standard = load_corpus("standard")
    assert quick.calls_per_run < standard.calls_per_run


# --- depth strata (§13.3) --------------------------------------------------


def test_standard_carries_the_documented_depth_ladder() -> None:
    assert load_corpus("standard").depth_strata() == {3: 4, 8: 4, 15: 4}


def test_every_depth_stratum_has_at_least_two_conversations() -> None:
    """Stratified resampling draws within a stratum; one member cannot vary."""
    for depth, count in load_corpus("standard").depth_strata().items():
        assert count >= 2, f"depth {depth} has only {count} conversation(s)"


def test_quick_has_a_single_depth_so_the_auc_degrades_to_one_point() -> None:
    """§10.2: a one-point curve has no area, and the report must say so."""
    assert list(load_corpus("quick").depth_strata()) == [3]


# --- structural guarantees -------------------------------------------------


def test_all_eight_attack_classes_are_present_at_standard() -> None:
    classes = {t.attack_class for t in load_corpus("standard").by_family("security")}
    assert classes == {
        "direct_injection",
        "instruction_override",
        "system_prompt_exfiltration",
        "role_confusion",
        "delimiter_escape",
        "indirect_injection",
        "scope_escalation",
        "cross_turn_injection",
    }


def test_all_five_policy_areas_at_all_four_pressure_levels() -> None:
    templates = load_corpus("standard").by_family("guardrail")
    cells = {(t.policy_id, t.pressure_level) for t in templates}
    policies = {"gr.pii", "gr.scope", "gr.regulated", "gr.disclosure", "gr.harm"}
    assert {p for p, _ in cells} == policies
    for policy in policies:
        assert {lvl for p, lvl in cells if p == policy} == {1, 2, 3, 4}


def test_every_attack_class_maps_to_owasp() -> None:
    for template in load_corpus("standard").by_family("security"):
        assert template.owasp, template.id
        assert template.owasp.startswith("LLM")


def test_the_hard_fail_classes_are_marked_critical() -> None:
    """D23: only critical classes short-circuit a config."""
    critical = {
        t.attack_class
        for t in load_corpus("standard").by_family("security")
        if t.severity == "critical"
    }
    assert "system_prompt_exfiltration" in critical
    assert "instruction_override" in critical


def test_cross_turn_probes_are_multi_turn() -> None:
    """A single-turn 'cross-turn' probe would test nothing."""
    for template in load_corpus("standard").by_family("security"):
        if template.attack_class == "cross_turn_injection":
            assert template.calls_per_run >= 3, template.id


def test_quick_carries_no_cross_turn_probes() -> None:
    """They cost three calls each and quick is sized for breadth, not depth."""
    classes = {t.attack_class for t in load_corpus("quick").by_family("security")}
    assert "cross_turn_injection" not in classes


def test_the_two_sampling_prompts_are_open_ended() -> None:
    """§9.1: a short factual prompt makes every target look INERT."""
    prompts = load_corpus("quick").instruments
    assert len(prompts) == 2
    for template in prompts:
        assert len(template.turns[0].text) > 80, template.id


def test_the_corpus_hash_is_profile_independent() -> None:
    """§10.3: two profiles of one corpus must agree it is the same corpus."""
    assert load_corpus("quick").hash == load_corpus("standard").hash
