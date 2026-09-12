"""Derived artifacts (spec §6.2, I7).

``aggregates.json`` and ``frontier.json`` *are* rewritten — that is what
"derived" means, and I7 is careful to say only raw observations are
append-only. The risk is a stale derived file being trusted after the log it
came from has grown, so every derived artifact carries a ``derived_from`` map
of ``{artifact: content_hash}`` and :func:`read_derived` reports staleness
rather than leaving the caller to guess.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from sweepeval.schema.versions import SCHEMA_VERSION, TOOL_VERSION

__all__ = ["DerivedArtifact", "read_derived", "write_derived"]

ModelT = TypeVar("ModelT", bound=BaseModel)


class DerivedArtifact(BaseModel):
    """Envelope written around every derived payload."""

    schema_version: str
    tool_version: str
    derived_from: dict[str, str]
    payload: dict[str, object]


def write_derived(
    path: Path,
    payload: BaseModel,
    *,
    derived_from: Mapping[str, str],
) -> None:
    """Write a derived artifact with its provenance."""
    envelope = DerivedArtifact(
        schema_version=SCHEMA_VERSION,
        tool_version=TOOL_VERSION,
        derived_from=dict(sorted(derived_from.items())),
        payload=payload.model_dump(mode="json"),
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
