"""The gate has to fail a build that deserves to fail (spec §16).

An independent audit found `sweepeval gate` exiting 0 on a run whose security
pass rate had gone from 100% to 0%, in three separate ways: a typo in
`--gate-on`, an opt-in to any of four objectives whose cluster tables are
keyed differently from their objective ids, and a security hard fail that
never reached the gate at all.

A green gate is the single most dangerous wrong answer this tool can give,
because it is the one nobody looks at twice.
"""

from __future__ import annotations

import pytest

from sweepeval.execute.gate import DEFAULT_GATE_ON
from sweepeval.schema.objective import REGISTRY, Objective
from sweepeval.stats.diff import ExitCode, gate_metrics


def _clusters(value: float, n: int = 24) -> dict[str, float]:
    return {f"p{i}": value for i in range(n)}


def _objective(
    metric: str,
    direction: str = "maximize",
    min_effect: float = 0.02,
    metric_key: str = "",
) -> Objective:
    return Objective(
        id=metric,
        display_label=metric,
        direction=direction,  # type: ignore[arg-type]
        family="security",
        cluster_key="probe",
        min_effect=min_effect,
        min_effect_kind="absolute",
        metric_key=metric_key,
    )


# --- a typo must not silently pass ----------------------------------------


def test_a_misspelled_gate_on_name_is_an_error_not_a_pass() -> None:
    """Measured: `--gate-on secuirty_pass_rate` exited 0 while security had
    gone from 100% to 0%."""
    objectives = [_objective("security_pass_rate")]
    with pytest.raises(ValueError, match="secuirty_pass_rate"):
        gate_metrics(
            objectives,
            {"security_pass_rate": _clusters(1.0)},
            {"security_pass_rate": _clusters(0.0)},
            gate_on=["secuirty_pass_rate"],
            seed=1,
        )


def test_the_error_lists_what_is_available() -> None:
    with pytest.raises(ValueError, match="security_pass_rate"):
        gate_metrics(
            [_objective("security_pass_rate")],
            {"security_pass_rate": _clusters(1.0)},
            {"security_pass_rate": _clusters(0.0)},
            gate_on=["nonsense"],
            seed=1,
        )


# --- objective ids are not always cluster keys ----------------------------


@pytest.mark.parametrize(
    ("objective_id", "cluster_key"),
    [("latency_mean_ms", "latency_ms"), ("cost_per_probe", "tokens_out")],
)
def test_opting_in_to_a_remapped_objective_actually_gates_it(
    objective_id: str, cluster_key: str
) -> None:
    """These two are stored under different keys from their ids. Looking up by
    id found nothing, so `--gate-on latency_p95_ms` gated nothing and said
    nothing about it."""
    objective = _objective(
        objective_id, direction="minimize", min_effect=10.0, metric_key=cluster_key
    )
    verdict = gate_metrics(
        [objective],
        {cluster_key: _clusters(300.0)},
        {cluster_key: _clusters(900.0)},
        gate_on=[objective_id],
        seed=1,
    )
    assert not verdict.ok, verdict.explain()
    assert verdict.exit_code is ExitCode.REGRESSION


# --- a gate that tested nothing is not a pass -----------------------------


def test_a_gate_with_no_shared_clusters_refuses_rather_than_passing() -> None:
    """Exiting 0 because every requested metric was missing tells the user
    their build is clean when nothing was checked. The runs shared no
    comparable data, which is exit 2 (§16)."""
    verdict = gate_metrics(
        [_objective("security_pass_rate")],
        {"security_pass_rate": _clusters(1.0)},
        {},
        seed=1,
    )
    assert not verdict.ok
    assert verdict.exit_code is ExitCode.COMPARABILITY_REFUSED
    assert any("no metric could be gated" in n for n in verdict.notes)


# --- the defaults cover what §16 says they cover --------------------------


def test_the_default_gate_covers_five_of_the_six_objectives() -> None:
    """Determinism, retention and cost were missing, so a regression in any of
    them never failed a default gate. The test that was here asserted only
    that latency was absent, which stayed green while four others were too."""
    defaults = {o.id for o in REGISTRY.defaults()}
    assert set(DEFAULT_GATE_ON) == defaults - {"latency_mean_ms"}


def test_latency_stays_out_of_the_default_gate() -> None:
    """Between-session variance is 20-50%; gating it flaps for reasons that
    have nothing to do with the code under test."""
    assert "latency_mean_ms" not in DEFAULT_GATE_ON


# --- a real regression still fires ----------------------------------------


def test_a_real_regression_fires() -> None:
    verdict = gate_metrics(
        [_objective("security_pass_rate")],
        {"security_pass_rate": _clusters(1.0)},
        {"security_pass_rate": _clusters(0.0)},
        seed=1,
    )
    assert not verdict.ok
    assert verdict.exit_code is ExitCode.REGRESSION
    assert verdict.regressions


def test_an_unchanged_run_passes() -> None:
    verdict = gate_metrics(
        [_objective("security_pass_rate")],
        {"security_pass_rate": _clusters(0.9)},
        {"security_pass_rate": _clusters(0.9)},
        seed=1,
    )
    assert verdict.ok
    assert verdict.exit_code is ExitCode.PASS


def test_a_hard_fail_fails_the_gate_on_its_own() -> None:
    """§16: any security hard fail fails, regardless of the baseline."""
    verdict = gate_metrics(
        [_objective("security_pass_rate")],
        {"security_pass_rate": _clusters(0.9)},
        {"security_pass_rate": _clusters(0.9)},
        hard_fails=("sec.exfiltration.direct.v1",),
        seed=1,
    )
    assert not verdict.ok
    assert verdict.hard_fails
