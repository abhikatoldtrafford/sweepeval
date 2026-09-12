"""``plan.json``, ``manifest.json`` and the resume check (spec §6.2, §12.5, I4)."""

from __future__ import annotations

from pathlib import Path

from sweepeval.corpus.loader import load_corpus
from sweepeval.execute.artifacts import (
    build_manifest,
    build_plan_document,
    read_json,
    verify_resume,
    write_json,
)
from sweepeval.execute.planner import ConfigSpec, SweepPlan
from sweepeval.execute.runner import RunPlan
from sweepeval.schema.comparability import Comparability, HardKeys, SoftKeys


def _plan(*, temperatures: tuple[float, ...] = (0.0, 0.7)) -> SweepPlan:
    configs = tuple(
        ConfigSpec(
            config_id=f"cfg-{i:02d}",
            params={"temperature": t},
            system_prompt=None,
            system_prompt_variant="none",
        )
        for i, t in enumerate(temperatures)
    )
    return SweepPlan(configs=configs, axes={"temperature": list(temperatures)}, cap=6)


def _units(n: int = 4):
    """Security units, because they are the ones that carry canaries."""
    corpus = load_corpus("quick")
    with_canaries = [t.to_unit() for t in corpus.probes if t.to_unit().canary_names]
    assert len(with_canaries) >= n, "the quick corpus lost its canary-bearing units"
    return tuple(with_canaries[:n])


def _document(
    *,
    run_id: str = "run-a",
    temperatures: tuple[float, ...] = (0.0, 0.7),
    units_n: int = 4,
    seed: str = "run-a:0",
):
    units = _units(units_n)
    canaries = RunPlan(
        config_id="_shared", units=units, runs=2, master_seed=seed
    ).canary_table()
    return build_plan_document(
        _plan(temperatures=temperatures),
        units,
        canaries,
        run_id=run_id,
        profile="quick",
        runs=2,
        master_seed=seed,
        corpus_hash="corpus-1",
    )


def _keys(**over) -> Comparability:
    hard = dict(
        schema_major=1,
        suite_version=1,
        corpus_hash="corpus-1",
        probe_layers=("generic",),
        target_type="model",
        similarity_backend="lexical",
        judge=None,
        profile="quick",
        pricing_source="none",
        scorer_versions={"security": 1},
        extraction_path="$.choices[0].message.content",
    )
    hard.update(over)
    return Comparability(
        hard=HardKeys(**hard),
        soft=SoftKeys(n_runs=2, concurrency=2, tool_version="0.1.0"),
    )


# --- I4 from the artifact --------------------------------------------------


def test_units_and_canaries_are_written_once_for_the_whole_sweep() -> None:
    """A per-config probe set would make an I4 violation representable."""
    payload = _document().to_dict()
    assert isinstance(payload["units"], list) and payload["units"]
    assert payload["canary_table"]
    for config in payload["configs"]:
        assert "units" not in config
        assert "canary_table" not in config


def test_canary_values_differ_across_runs() -> None:
    """So a cached response cannot pass by replaying an old canary (§11.2)."""
    table = _document().canary_table
    by_run: dict[str, set[str]] = {}
    for key, value in table.items():
        _unit_id, run_idx, _name = key.split("|")
        by_run.setdefault(run_idx, set()).add(value)
    assert len(by_run) >= 2
    runs = list(by_run.values())
    assert not runs[0] & runs[1], "a canary repeated across runs"


def test_the_plan_hash_ignores_the_run_it_belongs_to() -> None:
    """Otherwise a resume of the same experiment refuses itself."""
    assert _document(run_id="run-a", seed="run-a:0").hash() == _document(
        run_id="run-b", seed="run-b:0"
    ).hash()


def test_the_plan_hash_changes_when_the_sweep_changes() -> None:
    assert _document().hash() != _document(temperatures=(0.0, 0.7, 1.0)).hash()


def test_the_plan_hash_changes_when_the_probe_set_changes() -> None:
    assert _document().hash() != _document(units_n=5).hash()


# --- resume (§12.5) --------------------------------------------------------


def _manifest(**over):
    payload = dict(
        run_id="run-a",
        comparability=_keys(),
        plan_hash="plan-1",
        corpus_hash="corpus-1",
        capabilities={},
        target={"url": "https://x"},
        authorization=None,
        seed=0,
        master_seed="run-a:0",
        budget={},
        heuristics={},
    )
    payload.update(over)
    return build_manifest(**payload)


def test_an_unchanged_run_resumes() -> None:
    verdict = verify_resume(
        _manifest(), comparability=_keys(), plan_hash="plan-1", corpus_hash="corpus-1"
    )
    assert verdict.ok
    assert "every hard key match" in verdict.explain()


def test_an_edited_plan_refuses_rather_than_mixing_rows() -> None:
    verdict = verify_resume(
        _manifest(), comparability=_keys(), plan_hash="plan-2", corpus_hash="corpus-1"
    )
    assert not verdict.ok
    assert any(r.key == "plan_hash" for r in verdict.refusals)
    assert "would mix rows" in verdict.explain()


def test_a_changed_corpus_refuses() -> None:
    verdict = verify_resume(
        _manifest(), comparability=_keys(), plan_hash="plan-1", corpus_hash="corpus-2"
    )
    assert not verdict.ok
    assert any(r.key == "corpus_hash" for r in verdict.refusals)


def test_every_hard_key_is_verified_on_resume() -> None:
    """Not a spot check: a key added later must be covered automatically."""
    for key, changed in (
        ("target_type", "agent"),
        ("profile", "standard"),
        ("pricing_source", "user"),
        ("extraction_path", "$.other"),
        ("similarity_backend", "embedding"),
        ("suite_version", 2),
        ("schema_major", 2),
        ("probe_layers", ("generic", "user")),
        ("scorer_versions", {"security": 2}),
    ):
        verdict = verify_resume(
            _manifest(),
            comparability=_keys(**{key: changed}),
            plan_hash="plan-1",
            corpus_hash="corpus-1",
        )
        assert not verdict.ok, key
        assert any(r.key == key for r in verdict.refusals), key


def test_a_refusal_names_the_key_and_both_values() -> None:
    verdict = verify_resume(
        _manifest(),
        comparability=_keys(profile="standard"),
        plan_hash="plan-1",
        corpus_hash="corpus-1",
    )
    mismatch = next(r for r in verdict.refusals if r.key == "profile")
    assert "quick" in mismatch.message and "standard" in mismatch.message


def test_a_manifest_without_keys_cannot_be_resumed_against() -> None:
    """Absence is not a match."""
    verdict = verify_resume(
        {"plan_hash": "plan-1", "corpus_hash": "corpus-1"},
        comparability=_keys(),
        plan_hash="plan-1",
        corpus_hash="corpus-1",
    )
    assert not verdict.ok
    assert any(r.key == "comparability" for r in verdict.refusals)


def test_a_missing_manifest_refuses_rather_than_starting_fresh() -> None:
    verdict = verify_resume(
        {}, comparability=_keys(), plan_hash="plan-1", corpus_hash="corpus-1"
    )
    assert not verdict.ok


def test_artifacts_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_json(path, _manifest())
    assert read_json(path)["plan_hash"] == "plan-1"
    assert read_json(tmp_path / "absent.json") == {}


def test_a_torn_artifact_reads_as_absent_rather_than_crashing(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"plan_hash": "pla', encoding="utf-8")
    assert read_json(path) == {}


def test_the_manifest_names_the_token_heuristic() -> None:
    """§12.4: an estimate presented as a measurement is worse than none."""
    assert _manifest(heuristics={"tokens": "chars/4"})["heuristics"]["tokens"] == (
        "chars/4"
    )
