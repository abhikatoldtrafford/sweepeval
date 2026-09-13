"""Derived artifacts (spec §6.2, I7).

``aggregates.json`` and ``frontier.json`` *are* rewritten — that is what
"derived" means, and I7 is careful to say only raw observations are
append-only. The risk is a stale derived file being trusted after the log it
came from has grown, so every derived artifact carries a ``derived_from`` map
of ``{artifact: content_hash}`` and :func:`read_derived` reports staleness
rather than leaving the caller to guess.

For a long time this module was reachable only from its own unit tests: the
production writers went through :func:`sweepeval.store.json_io.write_json`,
so no ``aggregates.json`` or ``frontier.json`` the tool ever wrote carried a
``derived_from`` at all. I7's enforcement mechanism is named as "derived files
carry a `derived_from` hash", and it was not enforcing anything.

Envelopes are read leniently. Runs written before the writers were routed here
are flat payloads with no provenance, and refusing them would make an audit
trail change break offline re-reporting of exactly the runs an audit trail is
for. :func:`read_derived` returns an empty provenance map for those, and
staleness is reported as unknown rather than as false.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from sweepeval.schema.hashing import sha256_hex
from sweepeval.schema.versions import SCHEMA_VERSION, TOOL_VERSION

__all__ = [
    "DerivedArtifact",
    "provenance_of",
    "read_derived",
    "read_payload",
    "write_derived",
]

ModelT = TypeVar("ModelT", bound=BaseModel)


class DerivedArtifact(BaseModel):
    """Envelope written around every derived payload."""

    schema_version: str
    tool_version: str
    derived_from: dict[str, str]
    payload: dict[str, object]


def write_derived(
    path: Path,
    payload: BaseModel | Mapping[str, Any],
    *,
    derived_from: Mapping[str, str],
) -> None:
    """Write a derived artifact with its provenance.

    A plain mapping is accepted as well as a model because ``aggregates.json``
    is assembled as a dict -- and requiring a model here is why the production
    writers bypassed this function and the provenance was never written.
    """
    body = (
        payload.model_dump(mode="json")
        if isinstance(payload, BaseModel)
        else dict(payload)
    )
    envelope = DerivedArtifact(
        schema_version=SCHEMA_VERSION,
        tool_version=TOOL_VERSION,
        derived_from=dict(sorted(derived_from.items())),
        payload=body,
    )
    text = json.dumps(
        envelope.model_dump(mode="json"),
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def read_derived(
    path: Path,
    model: type[ModelT],
    *,
    current: Mapping[str, str] | None = None,
) -> tuple[ModelT, dict[str, str], bool]:
    """Read a derived artifact.

    Returns the payload, its recorded provenance, and whether it is stale with
    respect to ``current``. Staleness is returned rather than raised: ``report``
    legitimately reads a derived file whose log has since grown, and the right
    response is to regenerate, not to fail.
    """
    envelope = DerivedArtifact.model_validate_json(Path(path).read_text(encoding="utf-8"))
    stale = current is not None and dict(current) != envelope.derived_from
    return model.model_validate(envelope.payload), envelope.derived_from, stale


def read_payload(
    path: Path, *, current: Mapping[str, str] | None = None
) -> tuple[dict[str, Any], dict[str, str], bool | None]:
    """Read a derived artifact as a plain dict, envelope or not.

    Returns ``(payload, derived_from, stale)``. ``stale`` is ``None`` when it
    cannot be decided -- either no ``current`` map was supplied, or the file
    predates the envelope and carries no provenance to compare. Unknown is
    reported as unknown: a legacy file returned as "fresh" would be a claim
    the bytes do not support.

    An absent or torn file reads as ``({}, {}, None)``, matching
    :func:`sweepeval.store.json_io.read_json` -- a half-written aggregates
    file is not an aggregates file, and the caller's own "nothing stored here"
    path is the right one to take.
    """
    path = Path(path)
    if not path.exists():
        return {}, {}, None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, {}, None
    if not isinstance(loaded, dict):
        return {}, {}, None

    if _is_envelope(loaded):
        provenance = {
            str(k): str(v) for k, v in dict(loaded["derived_from"]).items()
        }
        payload = dict(loaded["payload"])
        stale = None if current is None else dict(current) != provenance
        return payload, provenance, stale

    return loaded, {}, None


def _is_envelope(loaded: Mapping[str, Any]) -> bool:
    """Distinguish an envelope from a flat payload.

    Both keys, and of the right types: ``payload`` alone is a plausible field
    name for a future flat document, and mistaking one for an envelope would
    silently return a fragment of it.
    """
    return (
        isinstance(loaded.get("payload"), dict)
        and isinstance(loaded.get("derived_from"), dict)
    )


def provenance_of(run_dir: Path) -> dict[str, str]:
    """The ``derived_from`` map for a run directory read back off disk.

    ``Store.provenance()`` is the same map for a live run. This one exists for
    ``sweepeval report``, which re-derives a frontier from a stored run it
    never executed and so holds no Store.
    """
    run_dir = Path(run_dir)
    return {
        name: sha256_hex(
            (run_dir / name).read_bytes() if (run_dir / name).exists() else b""
        )
        for name in ("calls.jsonl", "observations.jsonl")
    }
