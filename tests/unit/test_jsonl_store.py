"""Append-only JSONL log and derived artifacts (spec §6.2). I7 load-bearing."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from sweepeval.schema.call import Call
from sweepeval.store.derived import read_derived, write_derived
from sweepeval.store.jsonl import AppendOnlyLog
from sweepeval.store.redaction import Redactor


@pytest.fixture
def log(tmp_path: Path) -> AppendOnlyLog[Call]:
    return AppendOnlyLog(
        tmp_path / "calls.jsonl", Call, Redactor(secrets=["topsecret1"])
    )


# --- append and read ------------------------------------------------------


def test_append_then_read_roundtrips(log: AppendOnlyLog[Call]) -> None:
    log.append(Call.example())
    assert [c.unit_id for c in log.read()] == ["u#0123456789abcdef"]


def test_reading_a_missing_file_yields_nothing(tmp_path: Path) -> None:
    empty = AppendOnlyLog(tmp_path / "nope.jsonl", Call, Redactor())
    assert list(empty.read()) == []
    assert empty.count() == 0


def test_appends_accumulate_and_preserve_order(log: AppendOnlyLog[Call]) -> None:
    for idx in range(5):
        log.append(Call.example(run_idx=idx))
    assert [c.run_idx for c in log.read()] == [0, 1, 2, 3, 4]


def test_append_many_is_equivalent_to_repeated_append(log: AppendOnlyLog[Call]) -> None:
    log.append_many([Call.example(run_idx=0), Call.example(run_idx=1)])
    assert log.count() == 2


def test_append_many_with_no_rows_creates_nothing(tmp_path: Path) -> None:
    log: AppendOnlyLog[Call] = AppendOnlyLog(tmp_path / "c.jsonl", Call, Redactor())
    log.append_many([])
    assert not (tmp_path / "c.jsonl").exists()


def test_one_row_per_line(log: AppendOnlyLog[Call], tmp_path: Path) -> None:
    log.append_many([Call.example(run_idx=0), Call.example(run_idx=1)])
    assert len((tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()) == 2


# --- I7: no mutation API --------------------------------------------------


def test_there_is_no_mutation_api(log: AppendOnlyLog[Call]) -> None:
    public = {n for n in dir(log) if not n.startswith("_")}
    assert not (
        public
        & {"write", "update", "delete", "remove", "rewrite", "truncate", "clear", "seek"}
    )


def test_a_torn_final_line_does_not_poison_the_whole_file(
    log: AppendOnlyLog[Call], tmp_path: Path
) -> None:
    """A crash mid-append truncates the last row.

    That row's unit-run is not marked complete in the run state, so nothing
    aggregates it — but the preceding rows must still be readable, or one crash
    would destroy a whole config's paid-for results.
    """
    log.append(Call.example(run_idx=0))
    with (tmp_path / "calls.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"ts": "2026-09-12", "run_id": "r1", "conf')
    assert [c.run_idx for c in log.read()] == [0]


# --- redaction ------------------------------------------------------------


def test_rows_pass_through_the_redactor(log: AppendOnlyLog[Call], tmp_path: Path) -> None:
    log.append(Call.example(request=Call.example().request.model_copy(
        update={"shape": "shape-with-topsecret1-inside"}
    )))
    assert "topsecret1" not in (tmp_path / "calls.jsonl").read_text(encoding="utf-8")


# --- provenance -----------------------------------------------------------


def test_content_hash_changes_when_rows_are_appended(log: AppendOnlyLog[Call]) -> None:
    before = log.content_hash()
    log.append(Call.example())
    assert log.content_hash() != before


def test_content_hash_of_a_missing_file_is_stable(tmp_path: Path) -> None:
    a: AppendOnlyLog[Call] = AppendOnlyLog(tmp_path / "a.jsonl", Call, Redactor())
    b: AppendOnlyLog[Call] = AppendOnlyLog(tmp_path / "b.jsonl", Call, Redactor())
    assert a.content_hash() == b.content_hash()


# --- derived artifacts ----------------------------------------------------


class _Payload(BaseModel):
    value: int


def test_derived_roundtrips_with_its_provenance(tmp_path: Path) -> None:
    path = tmp_path / "aggregates.json"
    write_derived(path, _Payload(value=7), derived_from={"observations.jsonl": "abc"})
    payload, provenance, stale = read_derived(path, _Payload)
    assert payload.value == 7
    assert provenance == {"observations.jsonl": "abc"}
    assert stale is False


def test_derived_is_detected_as_stale_when_its_source_changed(tmp_path: Path) -> None:
    path = tmp_path / "aggregates.json"
    write_derived(path, _Payload(value=7), derived_from={"observations.jsonl": "abc"})
    _, _, stale = read_derived(path, _Payload, current={"observations.jsonl": "xyz"})
    assert stale is True


def test_derived_is_rewritable_unlike_the_log(tmp_path: Path) -> None:
    """I7 covers raw observations, not derived files (§6.2)."""
    path = tmp_path / "aggregates.json"
    write_derived(path, _Payload(value=1), derived_from={"o": "a"})
    write_derived(path, _Payload(value=2), derived_from={"o": "b"})
    payload, _, _ = read_derived(path, _Payload)
    assert payload.value == 2


def test_derived_leaves_no_temp_file(tmp_path: Path) -> None:
    write_derived(tmp_path / "x.json", _Payload(value=1), derived_from={})
    assert not list(tmp_path.glob("*.tmp"))
