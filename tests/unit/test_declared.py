"""Declared sweeps: corrections, axes, pricing, constraints (spec §4.1, §8.6)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sweepeval.capabilities.detect import (
    Capability,
    CapabilityReport,
    CapabilityResult,
    Support,
)
from sweepeval.capabilities.sampling import SamplingVerdict, Verdict
from sweepeval.execute.declared import load_declared
from sweepeval.execute.planner import plan_sweep


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "sweepeval.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _caps(system_prompt: bool = True) -> CapabilityReport:
    report = CapabilityReport()
    report.results[Capability.SYSTEM_PROMPT] = CapabilityResult(
        Capability.SYSTEM_PROMPT,
        Support.SUPPORTED if system_prompt else Support.UNSUPPORTED,
        "behavioural probe",
        "high",
    )
    return report


# --- corrections -----------------------------------------------------------


def test_a_corrected_extraction_path_is_read(tmp_path: Path) -> None:
    """The most common correction: the nonce oracle did not fire and the blind
    walk guessed a field."""
    path = _write(tmp_path, {"extraction": {"text_path": "$.result.completion"}})
    assert load_declared(path).text_path == "$.result.completion"


def test_the_annotations_discovery_wrote_are_not_read_back_as_settings(
    tmp_path: Path,
) -> None:
    """The emitted file is full of confidence and evidence keys. Reading one
    back as an instruction would let a stale annotation become a setting."""
    path = _write(
        tmp_path,
        {
            "extraction": {
                "text_path": "$.a",
                "confidence": "low",
                "method": "blind_walk",
                "evidence": {"score": 0.4},
            },
            "discovery": {"posts": 12, "models_seen": ["m1", "m2"]},
        },
    )
    declared = load_declared(path)
    assert declared.text_path == "$.a"
    assert not declared.axes
    assert declared.pricing is None


# --- declared axes ---------------------------------------------------------


def test_a_declared_axis_is_swept_even_when_the_sampling_test_said_inert(
    tmp_path: Path,
) -> None:
    """You asserted it matters; the tool takes your word."""
    declared = load_declared(_write(tmp_path, {"axes": {"temperature": [0.0, 0.9]}}))
    plan = plan_sweep(
        _caps(system_prompt=False),
        ["m1"],
        profile="standard",
        sampling={
            "temperature": SamplingVerdict(
                parameter="temperature", verdict=Verdict.INERT, tier=2, reason="TOST"
            )
        },
        declared_axes=declared.axes,
    )
    assert plan.axes["temperature"] == [0.0, 0.9]
    assert not any(a == "temperature" for a, _ in plan.rejected_axes), (
        "an axis the user declared must not also be listed as rejected"
    )


def test_a_declared_axis_overrides_the_discovered_values(tmp_path: Path) -> None:
    declared = load_declared(_write(tmp_path, {"axes": {"model": ["a", "b"]}}))
    plan = plan_sweep(
        _caps(system_prompt=False),
        ["discovered-1", "discovered-2", "discovered-3"],
        profile="standard",
        declared_axes=declared.axes,
    )
    assert plan.axes["model"] == ["a", "b"]


def test_a_one_value_axis_is_rejected_with_a_reason(tmp_path: Path) -> None:
    """It would put a column in plan.json that never varies, which reads as a
    swept dimension that found nothing."""
    declared = load_declared(_write(tmp_path, {"axes": {"model": ["only-one"]}}))
    assert not declared.axes
    assert any("at least 2" in w for w in declared.warnings)


def test_a_header_axis_is_accepted_and_flagged_as_unusual(tmp_path: Path) -> None:
    declared = load_declared(
        _write(tmp_path, {"axes": {"headers.x-retrieval-mode": ["off", "hybrid"]}})
    )
    assert declared.axes["headers.x-retrieval-mode"] == ["off", "hybrid"]
    assert not declared.warnings


def test_an_unknown_axis_name_warns_rather_than_failing(tmp_path: Path) -> None:
    declared = load_declared(_write(tmp_path, {"axes": {"frobnicate": [1, 2]}}))
    assert declared.axes["frobnicate"] == [1, 2]
    assert any("request-body field" in w for w in declared.warnings)


def test_a_header_axis_reaches_the_headers_not_the_body(tmp_path: Path) -> None:
    """A routing header swept as a body field is sent somewhere the target
    never reads, and the axis produces identical configs while looking like it
    swept something."""
    from sweepeval.execute.planner import ConfigSpec

    spec = ConfigSpec(
        config_id="cfg-00",
        params={"headers.x-mode": "hybrid", "temperature": 0.0},
        system_prompt=None,
        system_prompt_variant="none",
    )
    headers = {
        name[len("headers.") :]: str(value)
        for name, value in spec.params.items()
        if name.startswith("headers.")
    }
    body_params = {
        k: v for k, v in spec.params.items() if not k.startswith("headers.")
    }
    assert headers == {"x-mode": "hybrid"}
    assert body_params == {"temperature": 0.0}


# --- pricing ---------------------------------------------------------------


def test_pricing_turns_the_cost_objective_into_money(tmp_path: Path) -> None:
    declared = load_declared(
        _write(
            tmp_path,
            {
                "pricing": {
                    "input_per_mtok": 0.15,
                    "output_per_mtok": 0.60,
                    "currency": "USD",
                    "source": "our invoice",
                }
            },
        )
    )
    assert declared.pricing is not None
    assert declared.pricing.source == "our invoice"
    assert declared.pricing.cost(1_000_000, 0) == pytest.approx(0.15)


def test_malformed_pricing_is_ignored_with_a_warning(tmp_path: Path) -> None:
    """Silently treating it as zero would make cost look free."""
    declared = load_declared(_write(tmp_path, {"pricing": {"input_per_mtok": "free"}}))
    assert declared.pricing is None
    assert any("stays in tokens" in w for w in declared.warnings)


# --- constraints and objectives -------------------------------------------


def test_constraints_are_read(tmp_path: Path) -> None:
    declared = load_declared(
        _write(
            tmp_path,
            {"constraints": [{"metric": "error_rate", "limit": 0.02}]},
        )
    )
    assert declared.constraints[0].metric == "error_rate"
    assert declared.constraints[0].limit == 0.02
    assert declared.constraints[0].kind == "interval"


def test_a_malformed_constraint_is_ignored_with_a_warning(tmp_path: Path) -> None:
    declared = load_declared(
        _write(tmp_path, {"constraints": [{"metric": "error_rate"}]})
    )
    assert not declared.constraints
    assert any("numeric limit" in w for w in declared.warnings)


def test_profile_and_runs_are_read(tmp_path: Path) -> None:
    declared = load_declared(_write(tmp_path, {"profile": "standard", "runs": 5}))
    assert declared.profile == "standard"
    assert declared.runs == 5


def test_an_unknown_profile_warns_rather_than_being_used(tmp_path: Path) -> None:
    declared = load_declared(_write(tmp_path, {"profile": "exhaustive"}))
    assert declared.profile is None
    assert any("unknown profile" in w for w in declared.warnings)


def test_an_empty_config_declares_nothing(tmp_path: Path) -> None:
    declared = load_declared(_write(tmp_path, {"target": {"url": "https://x"}}))
    assert declared.url == "https://x"
    assert not declared.axes
    assert not declared.constraints
    assert not declared.warnings
    assert declared.describe() == []


def test_the_summary_names_every_override(tmp_path: Path) -> None:
    """The user has to be able to see that their config took effect."""
    declared = load_declared(
        _write(
            tmp_path,
            {
                "extraction": {"text_path": "$.a"},
                "axes": {"model": ["a", "b"]},
                "pricing": {"input_per_mtok": 1.0, "output_per_mtok": 2.0},
                "constraints": [{"metric": "error_rate", "limit": 0.02}],
                "objectives": ["security_pass_rate"],
            },
        )
    )
    summary = "\n".join(declared.describe())
    assert "$.a" in summary
    assert "axis model" in summary
    assert "pricing" in summary
    assert "constraint" in summary
    assert "objectives narrowed" in summary
