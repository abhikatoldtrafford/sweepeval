"""Property: the log is append-only (spec §6.2, I7).

I7 is the invariant that makes ``report`` work offline and makes a dead sweep
resumable. The property that matters is monotonicity: whatever was readable
before an append is still readable, unchanged, after it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from sweepeval.schema.call import Call
from sweepeval.store.jsonl import AppendOnlyLog
from sweepeval.store.redaction import Redactor

_batches = st.lists(st.lists(st.integers(0, 50), min_size=0, max_size=4), max_size=5)


@given(batches=_batches)
@settings(max_examples=40, deadline=None)
def test_appending_never_changes_existing_rows(batches: list[list[int]]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log: AppendOnlyLog[Call] = AppendOnlyLog(
            Path(tmp) / "calls.jsonl", Call, Redactor()
        )
        seen: list[int] = []
        for batch in batches:
            before = [c.run_idx for c in log.read()]
            log.append_many([Call.example(run_idx=i) for i in batch])
            after = [c.run_idx for c in log.read()]

            # The prefix is untouched, and exactly the appended rows are added.
            assert after[: len(before)] == before
            assert after[len(before) :] == batch

            seen.extend(batch)
            assert after == seen


@given(rows=st.lists(st.integers(0, 50), max_size=8))
@settings(max_examples=40, deadline=None)
def test_count_always_equals_the_number_of_rows_read(rows: list[int]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log: AppendOnlyLog[Call] = AppendOnlyLog(
            Path(tmp) / "calls.jsonl", Call, Redactor()
        )
        log.append_many([Call.example(run_idx=i) for i in rows])
        assert log.count() == len(rows) == len(list(log.read()))


@given(rows=st.lists(st.integers(0, 50), min_size=1, max_size=6))
@settings(max_examples=30, deadline=None)
def test_content_hash_is_stable_while_the_file_is_unchanged(rows: list[int]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log: AppendOnlyLog[Call] = AppendOnlyLog(
            Path(tmp) / "calls.jsonl", Call, Redactor()
        )
        log.append_many([Call.example(run_idx=i) for i in rows])
        assert log.content_hash() == log.content_hash()
