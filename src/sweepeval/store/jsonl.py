"""Append-only JSONL log (spec §6.2). I7 load-bearing.

I7: "Raw observations are append-only. Derived artifacts are regenerable from
them without contacting the endpoint."

The enforcement is that this class exposes append and read, and nothing else.
There is no update, no rewrite, no truncate, and no seek. A partially-executed
config therefore leaves orphan rows rather than a corrupted file, and
aggregation excludes them by consulting the run state's completed unit-runs
(§12.5) instead of by editing history.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel

from sweepeval.schema.hashing import sha256_hex
from sweepeval.store.redaction import Redactor

__all__ = ["AppendOnlyLog"]

ModelT = TypeVar("ModelT", bound=BaseModel)


class AppendOnlyLog(Generic[ModelT]):
    """One JSONL file of typed rows."""

    def __init__(self, path: Path, model: type[ModelT], redactor: Redactor) -> None:
        """
        Args:
            redactor: required. Rows carry URLs and error text, and this file
                is the one a user is most likely to attach to a bug report.
        """
        self._path = Path(path)
        self._model = model
        self._redactor = redactor

    @property
    def path(self) -> Path:
        return self._path

    def append(self, row: ModelT) -> None:
        self.append_many((row,))

    def append_many(self, rows: Iterable[ModelT]) -> None:
        payloads = [self._encode(row) for row in rows]
        if not payloads:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Mode "a" only. Every write is a whole line terminated by a newline,
        # so a crash mid-append truncates at most the final partial row, which
        # read() skips rather than failing the whole file on.
        with self._path.open("a", encoding="utf-8", newline="\n") as handle:
            for payload in payloads:
                handle.write(payload + "\n")

    def _encode(self, row: ModelT) -> str:
        redacted = self._redactor.mapping(row.model_dump(mode="json"))
        return json.dumps(redacted, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def read(self) -> Iterator[ModelT]:
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    # A torn final line from a crashed run. Skipping is correct:
                    # the row was never completed, and the run state does not
                    # mark its unit-run complete, so nothing aggregates it.
                    continue
                yield self._model.model_validate(payload)

    def count(self) -> int:
        return sum(1 for _ in self.read())

    def content_hash(self) -> str:
        """Digest of the file's bytes, for ``derived_from`` provenance."""
        if not self._path.exists():
            return sha256_hex(b"")
        return sha256_hex(self._path.read_bytes())
