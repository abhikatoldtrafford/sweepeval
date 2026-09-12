"""Run state and resume checkpoints (spec §12.5).

Checkpointing is per ``(config_id, unit_id, run_idx)``, not per config. A
config is roughly 490 calls at ``standard``; losing all of it because a crash
landed at 99% is unacceptable on work the user paid for.

Aggregation reads only completed unit-runs. That is how orphan rows from a
partially-executed unit-run are excluded without ever rewriting the append-only
log (I7) — the log keeps the rows, and the state decides which of them count.
"""

from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["RunState", "new_run_id"]


def new_run_id() -> str:
    """A sortable, unique run id: ``YYYYMMDDTHHMMSSmmm-xxxxxx``.

    Chronologically sortable so ``report`` can list runs newest-first without
    opening every manifest, and suffixed with randomness because two runs can
    start in the same millisecond.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")[:-3]
    return f"{stamp}-{secrets.token_hex(3)}"


class RunState:
    """Per-config checkpoint file: completed unit-runs and budget counters."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._data: dict[str, object] = self._load()

    def _load(self) -> dict[str, object]:
        if not self._path.exists():
            return {"completed": [], "requests": 0, "tokens": 0}
        return json.loads(self._path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self._data, sort_keys=True, indent=2), encoding="utf-8"
        )
        # Atomic: a crash during checkpointing must not leave a state file that
        # claims work is done when it is not, nor lose work that is.
        os.replace(tmp, self._path)

    # --- unit-run completion ---------------------------------------------

    @staticmethod
    def _token(config_id: str, unit_id: str, run_idx: int) -> str:
        return f"{config_id}\x1f{unit_id}\x1f{run_idx}"

    def mark_complete(self, config_id: str, unit_id: str, run_idx: int) -> None:
        completed = set(self._completed_tokens())
        completed.add(self._token(config_id, unit_id, run_idx))
        self._data["completed"] = sorted(completed)
        self._save()

    def is_complete(self, config_id: str, unit_id: str, run_idx: int) -> bool:
        return self._token(config_id, unit_id, run_idx) in set(self._completed_tokens())

    def completed_unit_runs(self, config_id: str) -> set[tuple[str, int]]:
        out: set[tuple[str, int]] = set()
        for token in self._completed_tokens():
            cfg, unit_id, run_idx = token.split("\x1f")
            if cfg == config_id:
                out.add((unit_id, int(run_idx)))
        return out

    def _completed_tokens(self) -> list[str]:
        value = self._data.get("completed", [])
        return [str(v) for v in value] if isinstance(value, list) else []

    # --- budget ------------------------------------------------------------

    def record_budget(self, *, requests: int, tokens: int) -> None:
        """Accumulate spend so ``--resume`` reconstructs the counter (§12.5)."""
        self._data["requests"] = int(self._data.get("requests", 0) or 0) + requests
        self._data["tokens"] = int(self._data.get("tokens", 0) or 0) + tokens
        self._save()

    def budget(self) -> tuple[int, int]:
        return (
            int(self._data.get("requests", 0) or 0),
            int(self._data.get("tokens", 0) or 0),
        )
