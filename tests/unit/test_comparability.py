"""Comparability keys (spec §6.5). I6 load-bearing."""

from __future__ import annotations

from typing import Any

from sweepeval.schema.comparability import (
    _MESSAGES,
    Comparability,
    HardKeys,
    JudgeKey,
    SoftKeys,
    compare_keys,
)


def _hard(**over: Any) -> HardKeys:
    base: dict[str, Any] = dict(
        schema_major=1,
        suite_version=1,
        corpus_hash="a" * 64,
        probe_layers=("generic",),
        target_type="BARE_MODEL",
        similarity_backend="lexical",
        judge=None,
        profile="standard",
        pricing_source="none",
        scorer_versions={"security": 1},
        extraction_path="$.choices[0].message.content",
    )
    base.update(over)
    return HardKeys(**base)


def _c(hard: dict[str, Any] | None = None, local: bool = False) -> Comparability:
    return Comparability(
        hard=_hard(**(hard or {})),
        soft=SoftKeys(n_runs=3, concurrency=2, tool_version="0.1.0"),
        local=local,
    )


# --- the key set ----------------------------------------------------------


def test_identical_keys_compare_ok() -> None:
    assert compare_keys(_c(), _c()).ok


def test_all_eleven_hard_keys_are_declared() -> None:
    assert set(HardKeys.model_fields) == {
        "schema_major",
        "suite_version",
        "corpus_hash",
        "probe_layers",
        "target_type",
        "similarity_backend",
        "judge",
        "profile",
        "pricing_source",
        "scorer_versions",
        "extraction_path",
    }


def test_every_hard_key_has_a_refusal_message() -> None:
    """§6.5: refusal always names the key. A new hard key without a message
    would fall back to a generic string, which is the behaviour the spec
    forbids."""
    assert set(_MESSAGES) == set(HardKeys.model_fields)


def test_every_hard_key_refuses_on_mismatch() -> None:
    """Parametrised over the field list, so adding a hard key is covered."""
    variants: dict[str, Any] = {
        "schema_major": 2,
        "suite_version": 2,
        "corpus_hash": "b" * 64,
        "probe_layers": ("generic", "user"),
        "target_type": "AGENT_SYSTEM",
        "similarity_backend": "embeddings:text-3",
        "judge": JudgeKey(model="m", prompt_version=1),
        "profile": "deep",
        "pricing_source": "file",
        "scorer_versions": {"security": 2},
        "extraction_path": "$.text",
    }
    assert set(variants) == set(HardKeys.model_fields)
    for key, value in variants.items():
        verdict = compare_keys(_c(), _c(hard={key: value}))
        assert not verdict.ok, key
        assert verdict.refusals[0].key == key


# --- the four rev-2 promotions --------------------------------------------


def test_profile_mismatch_refuses() -> None:
    verdict = compare_keys(_c(), _c(hard={"profile": "deep"}))
    assert not verdict.ok
    assert "retention depth weighting" in verdict.refusals[0].message


def test_scorer_version_mismatch_refuses() -> None:
    assert not compare_keys(_c(), _c(hard={"scorer_versions": {"security": 2}})).ok


def test_pricing_source_mismatch_refuses() -> None:
    assert not compare_keys(_c(), _c(hard={"pricing_source": "file"})).ok


def test_extraction_path_mismatch_refuses() -> None:
    assert not compare_keys(_c(), _c(hard={"extraction_path": "$.text"})).ok


# --- message quality ------------------------------------------------------


def test_refusal_message_names_the_key_and_both_values() -> None:
    verdict = compare_keys(_c(), _c(hard={"corpus_hash": "b" * 64}))
    message = verdict.refusals[0].message
    assert "corpus hash differs" in message
    assert "aaaa" in message
    assert "bbbb" in message


def test_no_refusal_is_a_bare_incomparable() -> None:
    for key, template in _MESSAGES.items():
        assert "{a}" in template and "{b}" in template, key
        assert len(template) > 40, key


def test_explain_renders_refusals_and_warnings() -> None:
    other = _c(hard={"profile": "deep"})
    other = other.model_copy(
        update={"soft": other.soft.model_copy(update={"n_runs": 5})}
    )
    text = compare_keys(_c(), other).explain()
    assert "REFUSED" in text
    assert "warning" in text


# --- soft keys and locality ----------------------------------------------


def test_soft_key_mismatch_warns_but_does_not_refuse() -> None:
    other = _c()
    other = other.model_copy(
        update={"soft": other.soft.model_copy(update={"n_runs": 5})}
    )
    verdict = compare_keys(_c(), other)
    assert verdict.ok
    assert verdict.warnings[0].key == "n_runs"


def test_local_and_non_local_refuse() -> None:
    assert not compare_keys(_c(), _c(local=True)).ok


def test_two_local_runs_compare_as_a_within_project_trend() -> None:
    """§6.5 permits this explicitly; refusing it would delete the use case."""
    verdict = compare_keys(_c(local=True), _c(local=True))
    assert verdict.ok
    assert any(w.key == "local" for w in verdict.warnings)


def test_scorer_versions_comparison_ignores_key_order() -> None:
    a = _c(hard={"scorer_versions": {"security": 1, "guardrail": 1}})
    b = _c(hard={"scorer_versions": {"guardrail": 1, "security": 1}})
    assert compare_keys(a, b).ok
