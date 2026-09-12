"""Run state and resume checkpoints (spec §12.5)."""

from __future__ import annotations

from pathlib import Path

from sweepeval.store.state import RunState, new_run_id


def test_run_ids_sort_chronologically() -> None:
    """Sortable by timestamp, so ``report`` lists runs newest-first without
    opening every manifest.

    Ties inside a single millisecond order by the random suffix, which is
    arbitrary but harmless — real runs are seconds apart. The property that
    matters is that the timestamp prefix never goes backwards.
    """
    stamps = [new_run_id().split("-")[0] for _ in range(20)]
    assert stamps == sorted(stamps)


def test_run_id_has_the_documented_shape() -> None:
    run_id = new_run_id()
    stamp, suffix = run_id.split("-")
    assert len(stamp) == 18 and stamp[8] == "T"
    assert len(suffix) == 6


def test_run_ids_are_unique() -> None:
    assert len({new_run_id() for _ in range(200)}) == 200


def test_unit_run_completion_is_tracked_per_triple(tmp_path: Path) -> None:
    """Not per config: a standard config is ~490 calls and losing it at 99%
    because the checkpoint was coarse is unacceptable."""
    state = RunState(tmp_path / "c1.json")
    state.mark_complete("c1", "u1", 0)
    assert state.is_complete("c1", "u1", 0)
    assert not state.is_complete("c1", "u1", 1)
    assert not state.is_complete("c1", "u2", 0)
    assert not state.is_complete("c2", "u1", 0)


def test_completion_survives_reload(tmp_path: Path) -> None:
    path = tmp_path / "c1.json"
    RunState(path).mark_complete("c1", "u1", 2)
    assert RunState(path).is_complete("c1", "u1", 2)


def test_completed_unit_runs_returns_the_set(tmp_path: Path) -> None:
    state = RunState(tmp_path / "c1.json")
    state.mark_complete("c1", "u1", 0)
    state.mark_complete("c1", "u2", 1)
    state.mark_complete("c2", "u9", 0)
    assert state.completed_unit_runs("c1") == {("u1", 0), ("u2", 1)}


def test_marking_the_same_unit_run_twice_is_idempotent(tmp_path: Path) -> None:
    state = RunState(tmp_path / "c1.json")
    state.mark_complete("c1", "u1", 0)
    state.mark_complete("c1", "u1", 0)
    assert state.completed_unit_runs("c1") == {("u1", 0)}


def test_unit_ids_containing_a_hash_survive_the_round_trip(tmp_path: Path) -> None:
    """unit_id is ``template#params_hash``; the token separator must not collide."""
    state = RunState(tmp_path / "c1.json")
    state.mark_complete("c1", "sec.injection.direct.v1#0123456789abcdef", 0)
    assert state.completed_unit_runs("c1") == {
        ("sec.injection.direct.v1#0123456789abcdef", 0)
    }


def test_budget_counters_accumulate_and_persist(tmp_path: Path) -> None:
    """§12.5: --resume reconstructs the budget rather than restarting it at zero."""
    path = tmp_path / "c1.json"
    state = RunState(path)
    state.record_budget(requests=10, tokens=1000)
    state.record_budget(requests=5, tokens=200)
    assert RunState(path).budget() == (15, 1200)


def test_a_fresh_state_starts_empty(tmp_path: Path) -> None:
    state = RunState(tmp_path / "new.json")
    assert state.budget() == (0, 0)
    assert state.completed_unit_runs("c1") == set()


def test_no_temp_file_is_left_behind(tmp_path: Path) -> None:
    RunState(tmp_path / "c1.json").mark_complete("c1", "u1", 0)
    assert not list(tmp_path.glob("*.tmp"))
