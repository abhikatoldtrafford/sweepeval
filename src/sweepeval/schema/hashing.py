"""Canonical hashing (spec §6.4, §6.5, §10.3, §12.5).

Every identity in the system is a hash: ``unit_id`` (§6.1), ``params_hash``
(§6.4), ``corpus_hash`` (§10.3), and the plan hash that ``--resume`` verifies
against (§12.5). Canonicalisation therefore has to be exact. If it drifts on
key order, float spelling or unicode, ``unit_id`` changes between runs, I4
stops being checkable, and every cross-run comparison is refused for a reason
that has nothing to do with the data.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "canonical_json",
    "corpus_hash",
    "hash_obj",
    "param_hash",
    "sha256_hex",
]


def _validate(obj: Any) -> None:
    """Reject values that JSON would encode ambiguously or lossily.

    Two classes of problem, both silent if unchecked:

    * non-finite floats — ``json`` writes ``NaN``/``Infinity``, which are not
      JSON and which no two decoders agree on;
    * non-string mapping keys — ``json`` coerces ``{1: x}`` to ``{"1": x}``,
      so ``{1: x}`` and ``{"1": x}`` would hash identically.
    """
    if isinstance(obj, float) and not math.isfinite(obj):
        raise ValueError(f"non-finite float is not hashable: {obj!r}")
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(
                    "mapping keys must be str to hash unambiguously, got "
                    f"{type(key).__name__}: {key!r}"
                )
            _validate(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _validate(value)


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON: sorted keys, no whitespace, UTF-8, no NaN/Inf.

    ``ensure_ascii=False`` keeps text as UTF-8 rather than ``\\uXXXX`` escapes,
    so the digest of a probe containing non-ASCII text does not depend on the
    encoder's escaping policy.
    """
    _validate(obj)
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """Lowercase hex SHA-256 of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def hash_obj(obj: Any) -> str:
    """Stable 64-char digest of any JSON-representable object."""
    return sha256_hex(canonical_json(obj))


def param_hash(params: Mapping[str, Any]) -> str:
    """16-char digest of a sampling-parameter set (§6.4).

    Truncated because it appears on every row of ``calls.jsonl`` and only has
    to distinguish the handful of configurations in one sweep, not resist
    collision attacks.
    """
    return hash_obj(dict(params))[:16]


def corpus_hash(
    template_bytes: Sequence[bytes],
    suite_version: int,
    profile_defs: Mapping[str, Any],
) -> str:
    """Digest over the probe corpus (§10.3).

    Each template is hashed *before* folding into the running digest. Updating
    the digest with raw bytes would alias: ``[b"ab"]`` and ``[b"a", b"b"]``
    would collide, so splitting one probe into two would leave the corpus hash
    unchanged and let two incomparable runs compare.

    Templates are sorted so the hash does not depend on filesystem ordering.
    """
    digest = hashlib.sha256()
    for blob in sorted(template_bytes):
        digest.update(sha256_hex(blob).encode("ascii"))
    digest.update(canonical_json({"suite_version": suite_version}))
    digest.update(canonical_json({"profiles": dict(profile_defs)}))
    return digest.hexdigest()
