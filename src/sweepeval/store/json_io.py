"""Reading and writing the run's metadata documents (spec §6.2).

``manifest.json``, ``plan.json``, ``frontier.json`` and ``aggregates.json``
are whole documents rather than append-only logs, so they do not go through
:mod:`sweepeval.store.jsonl`.

They live in ``store`` and not beside their writers because the reporters read
them too, and a reporter reaching into ``execute`` for a file reader drags the
whole request layer in behind it — which the layering contract catches, and
which is the right thing for it to catch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["read_json", "write_json"]


def write_json(path: Path, payload: Any) -> None:
    """Write a metadata artifact readably and deterministically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    """Read one, treating an absent or torn file as absent.

    A half-written manifest is not a manifest. Returning ``{}`` lets the
    resume check refuse for the reason it exists to refuse for — nothing to
    verify against — rather than crashing with a JSON error.
    """
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}
