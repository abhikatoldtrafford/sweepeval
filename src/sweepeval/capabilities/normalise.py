"""Output normalisation for distinctness comparisons (spec §9.1).

Raw string distinctness is useless on a real endpoint. A response that echoes a
request id, stamps a timestamp, or varies only in trailing whitespace is
"distinct" every single time, so a target that ignores temperature completely
would be declared ``EFFECTIVE`` and the sweep would spend its budget on an axis
that does nothing.

The same normaliser is used for the Jaccard dispersion in tier 2, so the two
tiers cannot disagree about what "the same output" means.
"""

from __future__ import annotations

import re

__all__ = ["distinct_count", "normalise", "token_jaccard"]

_MASKS: tuple[tuple[re.Pattern[str], str], ...] = (
    # ISO-ish timestamps
    (
        re.compile(
            r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?\b"
        ),
         "<ts>",
    ),
    # uuids
    (
        re.compile(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I
        ),
         "<uuid>",
    ),
    # provider request ids
    (re.compile(r"\b(?:req|chatcmpl|msg|run|call)[-_][A-Za-z0-9]{6,}\b"), "<reqid>"),
    # bare epoch seconds/millis
    (re.compile(r"\b1[6-9]\d{8,11}\b"), "<epoch>"),
    # long hex runs
    (re.compile(r"\b[0-9a-f]{16,}\b", re.I), "<hex>"),
)


def normalise(text: str) -> str:
    """Case-fold, collapse whitespace, and mask volatile identifiers."""
    out = text
    for pattern, replacement in _MASKS:
        out = pattern.sub(replacement, out)
    out = re.sub(r"\s+", " ", out)
    return out.strip().casefold()


def distinct_count(texts: list[str]) -> int:
    """Number of distinct normalised outputs."""
    return len({normalise(t) for t in texts})


def token_jaccard(a: str, b: str) -> float:
    """Jaccard similarity over normalised whitespace tokens.

    1.0 for identical, 0.0 for disjoint. The similarity backend for v0.1 is
    lexical (D12), so this is also what semantic stability uses.
    """
    left = set(normalise(a).split())
    right = set(normalise(b).split())
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
