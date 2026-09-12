"""``MetricValue`` — I3 enforced at construction (spec §6.4, §13.4).

I3: "Every metric carries an interval, or is explicitly marked
``NO_VALID_INTERVAL``. Never a bare point value."

The enforcement claim is that there is genuinely no other construction path,
because ``MetricValue`` is the only type an aggregate field accepts. These
tests pin that: a bare point is a ``ValidationError``, and ``method=none``
without the flag is too, so an implementer cannot quietly opt out.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sweepeval.schema.metric import CIMethod, Estimand, Flag, MetricSpec, MetricValue


def test_interval_metric_is_accepted() -> None:
    value = MetricValue(
        point=0.9,
        lo=0.8,
        hi=0.95,
        method=CIMethod.cluster_bootstrap,
        n_clusters=24,
        alpha=0.05,
        estimand=Estimand.generalization,
    )
    assert value.point == 0.9


def test_bare_point_value_is_rejected() -> None:
    with pytest.raises(ValidationError):
        MetricValue(
            point=0.9,
            method=CIMethod.cluster_bootstrap,
            n_clusters=24,
            alpha=0.05,
            estimand=Estimand.generalization,
        )


def test_only_lo_is_rejected() -> None:
    with pytest.raises(ValidationError):
        MetricValue(
            point=0.9,
            lo=0.8,
            method=CIMethod.cluster_bootstrap,
            n_clusters=24,
            alpha=0.05,
            estimand=Estimand.generalization,
        )


def test_method_none_requires_the_no_valid_interval_flag() -> None:
    with pytest.raises(ValidationError):
        MetricValue(
            point=0.9,
            method=CIMethod.none,
            n_clusters=3,
            alpha=0.05,
            estimand=Estimand.generalization,
        )


def test_method_none_with_the_flag_is_accepted_and_has_no_bounds() -> None:
    value = MetricValue(
        point=0.9,
        method=CIMethod.none,
        n_clusters=3,
        alpha=0.05,
        estimand=Estimand.generalization,
        flags=(Flag.NO_VALID_INTERVAL, Flag.LOW_N),
    )
    assert value.lo is None
    assert value.hi is None


def test_no_valid_interval_flag_forbids_bounds() -> None:
    with pytest.raises(ValidationError):
        MetricValue(
            point=0.9,
            lo=0.8,
            hi=0.95,
            method=CIMethod.none,
            n_clusters=3,
            alpha=0.05,
            estimand=Estimand.generalization,
            flags=(Flag.NO_VALID_INTERVAL,),
        )


def test_bounds_must_bracket_the_point() -> None:
    with pytest.raises(ValidationError):
        MetricValue(
            point=0.99,
            lo=0.8,
            hi=0.95,
            method=CIMethod.t,
            n_clusters=8,
            alpha=0.05,
            estimand=Estimand.conditional,
        )


def test_lo_must_not_exceed_hi() -> None:
    with pytest.raises(ValidationError):
        MetricValue(
            point=0.9,
            lo=0.95,
            hi=0.8,
            method=CIMethod.t,
            n_clusters=8,
            alpha=0.05,
            estimand=Estimand.conditional,
        )


def test_degenerate_interval_is_allowed() -> None:
    """lo == point == hi is legitimate: a rate of 1.0 over clusters that all passed."""
    value = MetricValue(
        point=1.0,
        lo=1.0,
        hi=1.0,
        method=CIMethod.cluster_bootstrap,
        n_clusters=24,
        alpha=0.05,
        estimand=Estimand.generalization,
    )
    assert value.lo == value.hi == value.point


def test_estimand_is_required() -> None:
    """§13.2: an interval whose estimand is unstated does not say what it covers."""
    with pytest.raises(ValidationError):
        MetricValue(
            point=0.9,
            lo=0.8,
            hi=0.95,
            method=CIMethod.t,
            n_clusters=8,
            alpha=0.05,
        )


def test_metric_value_is_frozen() -> None:
    value = MetricValue(
        point=0.9,
        lo=0.8,
        hi=0.95,
        method=CIMethod.t,
        n_clusters=8,
        alpha=0.05,
        estimand=Estimand.conditional,
    )
    with pytest.raises(ValidationError):
        value.point = 0.5  # type: ignore[misc]


def test_indicative_is_distinct_from_no_valid_interval() -> None:
    """Spec §6.4, rev 2.1a.

    INDICATIVE means the interval is valid but wide. Conflating it with
    NO_VALID_INTERVAL would let a report imply a number is unusable when it is
    merely imprecise, which is the whole distinction the quick profile rests on.
    """
    value = MetricValue(
        point=0.9,
        lo=0.55,
        hi=0.99,
        method=CIMethod.cluster_bootstrap,
        n_clusters=10,
        alpha=0.05,
        estimand=Estimand.generalization,
        flags=(Flag.INDICATIVE,),
    )
    assert value.lo is not None
    assert Flag.NO_VALID_INTERVAL not in value.flags


def test_flags_serialise_as_strings() -> None:
    value = MetricValue(
        point=0.9,
        method=CIMethod.none,
        n_clusters=3,
        alpha=0.05,
        estimand=Estimand.generalization,
        flags=(Flag.NO_VALID_INTERVAL,),
    )
    assert '"NO_VALID_INTERVAL"' in value.model_dump_json()


def test_metric_spec_declares_direction_and_cluster_key() -> None:
    spec = MetricSpec(
        metric="security_pass_rate",
        family="security",
        direction="maximize",
        unit="rate",
        cluster_key="security_probe",
    )
    assert spec.direction == "maximize"
    assert spec.cluster_key == "security_probe"


def test_metric_spec_rejects_an_unknown_direction() -> None:
    with pytest.raises(ValidationError):
        MetricSpec(
            metric="x",
            family="security",
            direction="optimise",  # type: ignore[arg-type]
            unit="rate",
            cluster_key="security_probe",
        )
