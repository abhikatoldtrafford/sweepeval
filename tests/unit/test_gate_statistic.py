"""The gate has to compare the statistic it names (spec §16, §13.3).

`rank/frontier.py` knew that `context_retention_auc` is a depth-weighted
trapezoid and `latency_p95_ms` a quantile -- its own comment reads "comparing
it as a mean tests a different quantity from the one reported". `stats/diff.py`
took the unweighted mean of everything, and `context_retention_auc` is in
`DEFAULT_GATE_ON`.

So a model that keeps shallow recall and forgets mid-conversation passed CI
silently, and the two numbers the gate printed beside the verdict were means
under the AUC's name. `Baseline` had no `strata` field, so the depth labels
the curve needs could not even reach the gate.

Two dispatches that disagree is the shape here, not one wrong formula --
which is why the fix is a single `stats.statistic.statistic_for` that both
callers go through, and why this file asserts on the *gate*, the side that
was wrong.
"""

from __future__ import annotations

import pytest

from sweepeval.execute.gate import DEFAULT_GATE_ON, gate_payload
from sweepeval.report.machine import gate_annotations
from sweepeval.schema.baseline import Baseline
from sweepeval.schema.comparability import Comparability, HardKeys, SoftKeys
from sweepeval.schema.objective import REGISTRY
from sweepeval.stats.diff import ExitCode, gate_metrics
from sweepeval.stats.statistic import statistic_for

AUC = REGISTRY.get("context_retention_auc")


def _comparability() -> Comparability:
    return Comparability(
        hard=HardKeys(
            schema_major=1, suite_version=1, corpus_hash="a" * 64,
            probe_layers=("generic",), target_type="BARE_MODEL",
            similarity_backend="lexical", judge=None, profile="standard",
            pricing_source="none", scorer_versions={"security": 1},
            extraction_path="$.choices[0].message.content",
        ),
        soft=SoftKeys(n_runs=3, concurrency=2, tool_version="0.1.0"),
    )


def _retention(keep_deep: bool, n: int = 24):
    """Recall that moves *between* depths.

    The unweighted mean is deliberately close on both sides while the
    depth-weighted area is far apart -- that separation is the whole finding,
    and a fixture where both move together would pass either way.
    """
    values, strata = {}, {}
    for i in range(n):
        depth = [3, 8, 15][i % 3]
        cid = f"conv{i}"
        strata[cid] = f"d{depth}"
        values[cid] = float(depth in (8, 15)) if keep_deep else float(depth == 3)
    return values, strata


# --- the statistic ---------------------------------------------------------


def test_the_gate_sees_the_regression_the_mean_hides() -> None:
    base, strata = _retention(keep_deep=True)
    current, _ = _retention(keep_deep=False)

    verdict = gate_metrics(
        [AUC], {"context_retention_auc": base}, {"context_retention_auc": current},
        gate_on=["context_retention_auc"],
        strata={"context_retention_auc": strata}, seed=7,
    )
    assert verdict.exit_code is ExitCode.REGRESSION
    assert verdict.diffs[0].regressed


def test_without_the_depth_labels_the_same_regression_escapes() -> None:
    """Not an aspiration -- the reason `Baseline.strata` had to be added. This
    is what every gate did before it, and it is why the test above is not
    vacuous."""
    base, _ = _retention(keep_deep=True)
    current, _ = _retention(keep_deep=False)

    verdict = gate_metrics(
        [AUC], {"context_retention_auc": base}, {"context_retention_auc": current},
        gate_on=["context_retention_auc"], strata=None, seed=7,
    )
    assert verdict.exit_code is ExitCode.PASS
    assert verdict.degraded == ("context_retention_auc",)


def test_the_printed_points_are_the_statistic_not_the_mean() -> None:
    """"0.5 -> 0.5 ok" for an AUC that had gone 0.79 -> 0.21 is not only the
    wrong verdict; it is two numbers that make it look justified."""
    base, strata = _retention(keep_deep=True)
    current, _ = _retention(keep_deep=False)
    ids = sorted(base)

    verdict = gate_metrics(
        [AUC], {"context_retention_auc": base}, {"context_retention_auc": current},
        gate_on=["context_retention_auc"],
        strata={"context_retention_auc": strata}, seed=7,
    )
    row = verdict.diffs[0]
    assert row.baseline_point == pytest.approx(
        statistic_for(AUC, base, strata).fn(ids)
    )
    assert row.current_point == pytest.approx(
        statistic_for(AUC, current, strata).fn(ids)
    )
    assert row.baseline_point != pytest.approx(sum(base.values()) / len(base))


def test_a_baseline_round_trips_its_strata() -> None:
    """The labels have to survive being written and committed, or the gate is
    back on the mean the next time CI runs."""
    _values, strata = _retention(keep_deep=True)
    original = Baseline(
        run_id="r", created_at="2026-09-13T00:00:00Z", config_id="c",
        comparability=_comparability(),
        metrics={}, clusters={}, strata={"context_retention_auc": strata},
    )
    revived = Baseline.model_validate_json(original.model_dump_json())
    assert revived.strata == {"context_retention_auc": strata}


def test_a_baseline_written_before_strata_existed_still_loads() -> None:
    """Refusing it would make an honesty fix break every committed baseline."""
    import json

    payload = json.loads(
        Baseline(
            run_id="r", created_at="2026-09-13T00:00:00Z", config_id="c",
            comparability=_comparability(),
            metrics={}, clusters={},
        ).model_dump_json()
    )
    payload.pop("strata", None)
    assert Baseline.model_validate(payload).strata == {}


def test_strata_in_an_unexpected_format_fall_back_rather_than_scoring_zero() -> None:
    """`auc_statistic` skips any label not shaped `d<int>` and returns 0.0 when
    that leaves nothing -- which is 0.0 on BOTH sides, reads as no change, and
    passes every gate. Tolerable while strata lived in memory; `Baseline.strata`
    puts them in a file people can hand-edit."""
    base, strata = _retention(keep_deep=True)
    current, _ = _retention(keep_deep=False)
    mangled = {c: label.removeprefix("d") for c, label in strata.items()}

    verdict = gate_metrics(
        [AUC], {"context_retention_auc": base}, {"context_retention_auc": current},
        gate_on=["context_retention_auc"],
        strata={"context_retention_auc": mangled}, seed=7,
    )
    row = verdict.diffs[0]
    assert row.baseline_point != 0.0, "both sides collapsed to zero and tied"
    assert verdict.degraded == ("context_retention_auc",)


# --- what the gate could not do ------------------------------------------


def _one_shared_metric():
    table = {f"u{i}": 1.0 for i in range(12)}
    return {"security_pass_rate": table}, {"security_pass_rate": dict(table)}


def test_a_gate_that_tested_one_of_five_says_so() -> None:
    """`no_data` was read only when EVERY metric was missing, so a partial
    miss was discarded: exit 0, no annotation, no JSON field, nothing on
    stdout. Four of five lost to unscorability is the common case -- guardrail
    coverage in the shipped scorecard run ranged 0/20 to 13/20."""
    base, current = _one_shared_metric()
    verdict = gate_metrics(
        REGISTRY.defaults(), base, current, gate_on=list(DEFAULT_GATE_ON), seed=7
    )

    assert [d.metric for d in verdict.diffs] == ["security_pass_rate"]
    assert set(verdict.not_gated) == set(DEFAULT_GATE_ON) - {"security_pass_rate"}
    assert verdict.incomplete
    assert "NOT GATED" in verdict.explain()


def test_the_json_payload_carries_what_was_not_tested() -> None:
    """A CI job parsing gate.json had no way to tell a run that tested five
    metrics from one that tested one."""
    base, current = _one_shared_metric()
    payload = gate_payload(
        gate_metrics(
            REGISTRY.defaults(), base, current, gate_on=list(DEFAULT_GATE_ON), seed=7
        )
    )
    assert payload["incomplete"] is True
    assert "guardrail_pass_rate" in payload["not_gated"]


def test_github_gets_a_warning_for_each_untested_metric() -> None:
    base, current = _one_shared_metric()
    text = gate_annotations(
        gate_payload(
            gate_metrics(
                REGISTRY.defaults(), base, current,
                gate_on=list(DEFAULT_GATE_ON), seed=7,
            )
        )
    )
    assert text.count("::warning title=sweepeval not gated::") == 4


def test_a_complete_gate_reports_nothing_missing() -> None:
    """Otherwise the field is decoration that always fires."""
    base, current = _one_shared_metric()
    verdict = gate_metrics(
        REGISTRY.defaults(), base, current, gate_on=["security_pass_rate"], seed=7
    )
    assert verdict.not_gated == ()
    assert verdict.degraded == ()
    assert not verdict.incomplete
