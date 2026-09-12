"""Content-addressed blob store (spec §6.3, D37). I7 load-bearing.

Rev 1 stored only hashes, which made four promised features impossible:
showing the response that hard-failed a config (§11.2), rebuilding a report
offline (§5.1), semantic stability (§11.4), and judge escalation (§11.9). It
also meant a scorer bugfix required re-buying the entire dataset, on a tool
whose default run costs real money.

Content addressing dedupes identical responses for free, which Task 8.7 reuses
for cache detection (§12.7): the same body hash across runs is exactly the
signal.

There is deliberately no delete or overwrite API. The name *is* the content, so
content is immutable by construction rather than by convention.
"""

from __future__ import annotations

import os
from pathlib import Path

from sweepeval.schema.hashing import sha256_hex
from sweepeval.store.redaction import Redactor

__all__ = ["DEFAULT_CAP_BYTES", "BlobStore"]

DEFAULT_CAP_BYTES = 64 * 1024
_TRUNCATION_MARKER = b"\n[truncated by sweepeval]"


class BlobStore:
    """Immutable, deduped, redacted storage for response text and bodies."""

    def __init__(
        self,
        root: Path,
        redactor: Redactor,
        *,
        store_text: bool = True,
        cap_bytes: int = DEFAULT_CAP_BYTES,
    ) -> None:
        """
        Args:
            redactor: required, not optional. Every byte written passes through
                it, and making it a constructor argument is what stops a future
                writer from quietly bypassing it (§6.6).
            store_text: ``False`` under ``--no-store-bodies``. Suppresses
                ordinary text but never ``always=True`` content.
            cap_bytes: ordinary text over this is truncated, not dropped.
        """
        self._root = Path(root)
        self._redactor = redactor
        self._store_text = store_text
        self._cap_bytes = cap_bytes

    def put_text(self, text: str, *, always: bool = False) -> str | None:
        """Store text, returning its blob id, or ``None`` if suppressed.

        ``always=True`` is for evidence that the product cannot function
        without: discovery transcripts, error bodies, and hard-fail hits. It
        bypasses both the cap and ``--no-store-bodies`` — without that, the
        hard-fail report has nothing to show for an irreversible, build-
        breaking decision (§11.2).
        """
        return self.put_bytes(self._redactor.text(text).encode("utf-8"), always=always)

    def put_bytes(self, data: bytes, *, always: bool = False) -> str | None:
        if not always:
            if not self._store_text:
                return None
            if len(data) > self._cap_bytes:
                data = data[: self._cap_bytes] + _TRUNCATION_MARKER

        blob_id = sha256_hex(data)
        path = self._path(blob_id)
        if path.exists():
            return blob_id

        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a crash mid-write must not leave a partial blob
        # under a name that asserts its content hash.
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        return blob_id

    def get(self, blob_id: str) -> bytes:
        path = self._path(blob_id)
        if not path.exists():
            raise KeyError(f"no blob {blob_id!r}")
        return path.read_bytes()

    def exists(self, blob_id: str) -> bool:
        return self._path(blob_id).exists()

    def __contains__(self, blob_id: object) -> bool:
        return isinstance(blob_id, str) and self.exists(blob_id)

    def _path(self, blob_id: str) -> Path:
        # Sharded by prefix: a standard sweep writes thousands of blobs, and a
        # single flat directory degrades badly on most filesystems.
        return self._root / blob_id[:2] / blob_id
