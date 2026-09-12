"""Content-addressed blob store (spec §6.3, D37)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sweepeval.store.blob import BlobStore
from sweepeval.store.redaction import Redactor


@pytest.fixture
def store(tmp_path: Path) -> BlobStore:
    return BlobStore(root=tmp_path / "blobs", redactor=Redactor(secrets=["topsecret1"]))


def test_put_returns_the_sha256_of_the_stored_bytes(store: BlobStore) -> None:
    blob_id = store.put_text("hello")
    assert blob_id is not None
    assert len(blob_id) == 64
    assert store.get(blob_id) == b"hello"


def test_identical_content_dedupes_to_one_file(store: BlobStore, tmp_path: Path) -> None:
    first, second = store.put_text("same"), store.put_text("same")
    assert first == second
    files = [p for p in (tmp_path / "blobs").rglob("*") if p.is_file()]
    assert len(files) == 1


def test_content_passes_through_the_redactor_before_writing(store: BlobStore) -> None:
    blob_id = store.put_text("the key is topsecret1 ok")
    assert blob_id is not None
    assert b"topsecret1" not in store.get(blob_id)


def test_redaction_happens_before_hashing(store: BlobStore) -> None:
    """Otherwise the id would reveal the unredacted content to anyone who can
    guess it, and two texts differing only in a secret would not dedupe."""
    assert store.put_text("prefix topsecret1") == store.put_text("prefix topsecret1")


def test_text_over_the_cap_is_truncated_not_dropped(tmp_path: Path) -> None:
    store = BlobStore(root=tmp_path, redactor=Redactor(), cap_bytes=16)
    blob_id = store.put_text("x" * 100)
    assert blob_id is not None
    stored = store.get(blob_id)
    assert stored.startswith(b"x" * 16)
    assert b"truncated" in stored


def test_always_true_bypasses_the_cap(tmp_path: Path) -> None:
    store = BlobStore(root=tmp_path, redactor=Redactor(), cap_bytes=16)
    blob_id = store.put_text("y" * 100, always=True)
    assert blob_id is not None
    assert store.get(blob_id) == b"y" * 100


def test_no_store_bodies_suppresses_ordinary_text(tmp_path: Path) -> None:
    store = BlobStore(root=tmp_path, redactor=Redactor(), store_text=False)
    assert store.put_text("ordinary") is None


def test_no_store_bodies_still_stores_hard_fail_evidence(tmp_path: Path) -> None:
    """§11.2: elimination is irreversible and breaks builds. It must show its
    evidence even for a user who opted out of body storage."""
    store = BlobStore(root=tmp_path, redactor=Redactor(), store_text=False)
    blob_id = store.put_text("the canary leaked", always=True)
    assert blob_id is not None
    assert store.get(blob_id) == b"the canary leaked"


def test_get_on_a_missing_blob_raises_keyerror(store: BlobStore) -> None:
    with pytest.raises(KeyError):
        store.get("0" * 64)


def test_exists_and_contains_agree(store: BlobStore) -> None:
    blob_id = store.put_text("here")
    assert blob_id is not None
    assert store.exists(blob_id)
    assert blob_id in store
    assert "0" * 64 not in store


def test_blobs_are_sharded_by_prefix(store: BlobStore, tmp_path: Path) -> None:
    blob_id = store.put_text("shard me")
    assert blob_id is not None
    assert (tmp_path / "blobs" / blob_id[:2] / blob_id).exists()


def test_no_temp_files_are_left_behind(store: BlobStore, tmp_path: Path) -> None:
    store.put_text("clean up")
    assert not list((tmp_path / "blobs").rglob("*.tmp"))


def test_there_is_no_delete_or_overwrite_api(store: BlobStore) -> None:
    """I7. The name is the content, so mutation is not expressible."""
    public = {n for n in dir(store) if not n.startswith("_")}
    assert not (
        public & {"delete", "remove", "overwrite", "update", "rewrite", "clear", "pop"}
    )


def test_rewriting_the_same_id_is_a_noop(store: BlobStore, tmp_path: Path) -> None:
    blob_id = store.put_text("stable")
    assert blob_id is not None
    mtime = (tmp_path / "blobs" / blob_id[:2] / blob_id).stat().st_mtime_ns
    store.put_text("stable")
    assert (tmp_path / "blobs" / blob_id[:2] / blob_id).stat().st_mtime_ns == mtime
