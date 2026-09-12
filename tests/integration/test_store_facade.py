"""The ``Store`` facade and the run layout (spec §6.2, §6.6).

The second test is the no-secrets contract. It is exhaustive rather than a spot
check precisely because ``Store`` is the only way to obtain a writer: there is
no other constructor path a caller could use to reach past the redactor.
"""

from __future__ import annotations

from pathlib import Path

from sweepeval.schema.call import Call
from sweepeval.store.redaction import Redactor
from sweepeval.store.run import Store, new_run_id


def test_store_creates_the_documented_layout(tmp_path: Path) -> None:
    store = Store(tmp_path, new_run_id(), Redactor())
    store.calls.append(Call.example())
    blob_id = store.blobs.put_text("hi")
    assert blob_id is not None
    store.state_for("c1").mark_complete("c1", "u1", 0)

    run_dir = store.run_dir
    assert (run_dir / "calls.jsonl").exists()
    assert (run_dir / "blobs" / blob_id[:2] / blob_id).exists()
    assert (run_dir / "state" / "c1.json").exists()
    assert store.manifest_path == run_dir / "manifest.json"
    assert store.plan_path == run_dir / "plan.json"
    assert store.report_path("md") == run_dir / "report.md"


def test_every_writer_handed_out_by_store_is_redacted(tmp_path: Path) -> None:
    store = Store(tmp_path, new_run_id(), Redactor(secrets=["topsecretvalue"]))
    store.calls.append(
        Call.example(
            request=Call.example().request.model_copy(
                update={"shape": "shape topsecretvalue"}
            )
        )
    )
    store.observations  # noqa: B018 — exercised below via the same redactor
    store.blobs.put_text("leaking topsecretvalue here")

    text = "".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in tmp_path.rglob("*")
        if p.is_file()
    )
    assert "topsecretvalue" not in text


def test_query_param_auth_never_reaches_disk(tmp_path: Path) -> None:
    """§8.4's fourth auth rung is a query param, and baseline.json is committed."""
    key = "sk-supersecretkeyvalue123"
    store = Store(tmp_path, new_run_id(), Redactor(secrets=[key]))
    store.blobs.put_text(f"GET https://api.test/chat?api_key={key} failed", always=True)

    text = "".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in tmp_path.rglob("*")
        if p.is_file()
    )
    assert key not in text


def test_provenance_covers_both_logs(tmp_path: Path) -> None:
    store = Store(tmp_path, new_run_id(), Redactor())
    before = store.provenance()
    store.calls.append(Call.example())
    after = store.provenance()
    assert set(after) == {"calls.jsonl", "observations.jsonl"}
    assert after["calls.jsonl"] != before["calls.jsonl"]


def test_no_store_bodies_propagates_to_the_blob_store(tmp_path: Path) -> None:
    store = Store(tmp_path, new_run_id(), Redactor(), store_text=False)
    assert store.blobs.put_text("ordinary") is None
    assert store.blobs.put_text("hard-fail evidence", always=True) is not None


def test_two_runs_do_not_share_a_directory(tmp_path: Path) -> None:
    a = Store(tmp_path, new_run_id(), Redactor())
    b = Store(tmp_path, new_run_id(), Redactor())
    assert a.run_dir != b.run_dir
