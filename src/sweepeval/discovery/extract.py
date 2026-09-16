"""Response-text extraction inference (spec §8.5, D6).

**The nonce oracle is primary.** Send one probe asking the target to reply with
exactly a high-entropy nonce, then select the JSON path whose string value
contains it. One request, deterministic, and immune to the failure that sinks
the heuristic.

That failure is worth stating, because an earlier revision shipped the
heuristic alone. The blind walk scores candidate paths on length, variation
with input, and presence in every probe. An **echoed input field** —
``$.prompt``, ``$.echo``, ``$.messages[0].content``, or any framework's
request-mirroring debug field — maximises all three by construction: it is
long, it varies perfectly with input by definition, and it is always present.
On a target whose answers are short, the echo wins outright, and every
text-derived metric is then computed on the tool's own prompt.

The oracle also defeats a proxy that mimics a known envelope while putting the
real text elsewhere, which family priors cannot.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "SIBLING_REFUSAL_FIELDS",
    "STOPLIST",
    "ExtractionCandidate",
    "ExtractionResult",
    "extract_at",
    "extract_text",
    "infer_delta_path",
    "infer_from_nonce",
    "infer_from_walk",
    "walk_string_paths",
]

STOPLIST = frozenset(
    {
        # metadata that is never the answer
        "id",
        "model",
        "role",
        "type",
        "object",
        "created",
        "finish_reason",
        "finishreason",
        "stop_reason",
        "index",
        "usage",
        "system_fingerprint",
        # error envelopes — verbose, input-varying, and not the answer
        "error",
        "detail",
        "message",
        "warning",
        "code",
        # request mirrors — the failure mode the oracle exists to defeat
        "prompt",
        "input",
        "echo",
        "request",
        "request_text",
        "query",
        "messages",
        "contents",
    }
)

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_BASE64ISH = re.compile(r"^[A-Za-z0-9+/=_-]{16,}$")
_ENUMISH = re.compile(r"^[a-z][a-z0-9_]{0,24}$")


@dataclass(frozen=True)
class ExtractionCandidate:
    path: str
    score: float
    reasons: tuple[str, ...] = ()


@dataclass
class ExtractionResult:
    """An inferred extraction path plus the evidence for it."""

    path: str | None
    confidence: str
    method: str
    evidence: dict[str, Any] = field(default_factory=dict)
    candidates: tuple[ExtractionCandidate, ...] = ()

    @property
    def ok(self) -> bool:
        return self.path is not None


# --- walking --------------------------------------------------------------


def walk_string_paths(payload: Any, prefix: str = "$") -> list[tuple[str, str]]:
    """Every ``(jsonpath, string value)`` in a decoded body."""
    found: list[tuple[str, str]] = []

    if isinstance(payload, str):
        found.append((prefix, payload))
    elif isinstance(payload, dict):
        for key, value in payload.items():
            found.extend(walk_string_paths(value, f"{prefix}.{key}"))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(walk_string_paths(value, f"{prefix}[{index}]"))

    return found


def extract_at(payload: Any, path: str) -> str | None:
    """Read a value at a path produced by :func:`walk_string_paths`.

    Also supports the ``[*]`` wildcard used by Anthropic's ``$.content[*].text``,
    joining matches in document order with no separator (§8.5).
    """
    if path == "$":
        return payload if isinstance(payload, str) else None

    tokens = _tokenise(path)
    values = _resolve(payload, tokens)
    strings = [v for v in values if isinstance(v, str)]
    if not strings:
        return None
    return "".join(strings)


SIBLING_REFUSAL_FIELDS: tuple[str, ...] = ("refusal", "refusal_text")
"""Fields a provider may use *instead of* the content field, for a refusal.

OpenAI returns ``content: null`` with ``refusal`` populated when the model
takes its structured refusal path. Discovery finds the content field, because
that is where text lives in every response it probed with -- inert prompts do
not get refused -- so a refusal extracts to nothing.

The cost lands in the worst place: a refusal is the *correct* answer to an
injection probe, so the trials lost are the ones where the target behaved
best. Five of gpt-5.1's 72 security trials went UNSCORABLE in a live run for
exactly this.
"""


def extract_text(payload: Any, path: str | None) -> str:
    """Text at ``path``, falling back to a refusal beside it.

    Lives here rather than in the runner because it had exactly one caller and
    needed two. The runner learned to read a structured refusal after a live
    run lost five trials to one; the capability detector did not, so
    `detect_refusal_baseline` -- whose entire job is recognising how a target
    declines -- was blind to the most structured way of declining there is.
    The sibling is only consulted when the content field is empty, so a target
    that answers normally is unaffected.
    """
    if payload is None or not path:
        return ""
    text = extract_at(payload, path) or ""
    if text:
        return text

    parent, _, _ = path.rpartition(".")
    if not parent:
        return ""
    for name in SIBLING_REFUSAL_FIELDS:
        found = extract_at(payload, f"{parent}.{name}")
        if found:
            return found
    return ""


def _tokenise(path: str) -> list[str | int]:
    tokens: list[str | int] = []
    for part in path.lstrip("$").split("."):
        if not part:
            continue
        while "[" in part:
            name, _, rest = part.partition("[")
            if name:
                tokens.append(name)
            index, _, part = rest.partition("]")
            tokens.append("*" if index == "*" else int(index))
        if part:
            tokens.append(part)
    return tokens


def _resolve(payload: Any, tokens: Sequence[str | int]) -> list[Any]:
    current: list[Any] = [payload]
    for token in tokens:
        nxt: list[Any] = []
        for node in current:
            if token == "*":
                if isinstance(node, list):
                    nxt.extend(node)
            elif isinstance(token, int):
                if isinstance(node, list) and 0 <= token < len(node):
                    nxt.append(node[token])
            elif isinstance(node, dict) and token in node:
                nxt.append(node[token])
        current = nxt
        if not current:
            return []
    return current


# --- the oracle -----------------------------------------------------------


def infer_from_nonce(payload: Any, nonce: str) -> ExtractionResult:
    """Primary method: find the path whose value contains the nonce.

    Failure criterion, which the plan flagged as undefined: the oracle fails if
    no string value contains the nonce, or if several distinct paths contain it
    and they still disagree after preferring the deepest. Which of the two is
    recorded, so the report can say why it fell back.
    """
    matches = [
        (path, value)
        for path, value in walk_string_paths(payload)
        if nonce in value
    ]

    if not matches:
        return ExtractionResult(
            path=None,
            confidence="none",
            method="nonce_oracle",
            evidence={"nonce_found": False, "reason": "no value contained the nonce"},
        )

    if len(matches) == 1:
        path, value = matches[0]
        return ExtractionResult(
            path=path,
            confidence="high",
            method="nonce_oracle",
            evidence={
                "nonce_found_at": path,
                "value_len": len(value),
                "unique_match": True,
            },
        )

    # Several paths carry the nonce — typically the answer plus an echo of the
    # request that asked for it. Prefer the deepest path, then the shortest
    # value: the answer is "X7K2Q9", the echo is the whole instruction.
    ranked = sorted(matches, key=lambda pv: (-pv[0].count("."), len(pv[1])))
    best_path, best_value = ranked[0]
    tie = len(ranked) > 1 and ranked[1][0].count(".") == best_path.count(".")

    return ExtractionResult(
        path=best_path,
        confidence="medium" if tie else "high",
        method="nonce_oracle",
        evidence={
            "nonce_found_at": best_path,
            "value_len": len(best_value),
            "unique_match": False,
            "other_matches": [p for p, _ in ranked[1:]],
            "reason": (
                "several paths carried the nonce; preferred the deepest, then "
                "the shortest value"
            ),
        },
    )


# --- the fallback walk ----------------------------------------------------


def infer_from_walk(
    payloads: Sequence[Any],
    prompts: Sequence[str],
    priors: Iterable[str] = (),
) -> ExtractionResult:
    """Fallback: score candidate paths across several probes.

    Only used when the oracle fails — the target would not comply, or wrapped
    the nonce. Its scoring is the rev-1 heuristic plus the three corrections
    that make the echo lose: a near-duplicate-of-request penalty, an extended
    stoplist, and a preference for paths present in every probe.
    """
    if not payloads:
        return ExtractionResult(
            path=None, confidence="none", method="walk", evidence={"reason": "no probes"}
        )

    per_probe = [dict(walk_string_paths(p)) for p in payloads]
    all_paths = sorted({path for probe in per_probe for path in probe})
    prior_set = set(priors)

    lengths = [
        len(value)
        for probe in per_probe
        for value in probe.values()
    ] or [1]
    max_len = max(lengths)

    candidates: list[ExtractionCandidate] = []
    for path in all_paths:
        values = [probe.get(path) for probe in per_probe]
        present = [v for v in values if v is not None]
        if not present:
            continue

        reasons: list[str] = []
        score = 0.0

        presence = len(present) / len(per_probe)
        score += 2.0 * presence
        if presence == 1.0:
            reasons.append("present in every probe")

        varies = len({v for v in present}) > 1
        if varies:
            score += 2.0
            reasons.append("varies with input")

        mean_len = sum(len(v) for v in present) / len(present)
        score += 2.0 * (mean_len / max_len)

        leaf = path.rsplit(".", 1)[-1].split("[")[0].lower()
        if leaf in STOPLIST:
            score -= 6.0
            reasons.append(f"stoplisted key {leaf!r}")

        if all(_looks_like_metadata(v) for v in present):
            score -= 3.0
            reasons.append("value looks like an id or enum")

        # The correction that matters: a value that is (nearly) the request is
        # an echo, not an answer.
        echo = _echo_overlap(present, prompts)
        if echo >= 0.9:
            score -= 8.0
            reasons.append(f"near-duplicate of the request ({echo:.0%})")

        if path in prior_set:
            score += 3.0
            reasons.append("matches a family prior")

        candidates.append(
            ExtractionCandidate(path=path, score=score, reasons=tuple(reasons))
        )

    if not candidates:
        return ExtractionResult(
            path=None, confidence="none", method="walk", evidence={"reason": "no candidates"}
        )

    ranked = sorted(candidates, key=lambda c: (-c.score, c.path))
    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    margin = best.score - runner_up.score if runner_up else best.score

    confidence = "high" if margin >= 3.0 else "medium" if margin >= 1.0 else "low"

    return ExtractionResult(
        path=best.path,
        confidence=confidence,
        method="walk",
        evidence={
            "walk_top1": f"{best.path} ({best.score:.1f})",
            "walk_top2": (
                f"{runner_up.path} ({runner_up.score:.1f})" if runner_up else None
            ),
            "margin": round(margin, 2),
            "reasons": list(best.reasons),
        },
        candidates=tuple(ranked[:5]),
    )


def _looks_like_metadata(value: str) -> bool:
    return bool(_UUID.match(value) or (_ENUMISH.match(value) and len(value) < 25))


def _echo_overlap(values: Sequence[str], prompts: Sequence[str]) -> float:
    """How much of the request the value reproduces, at its worst."""
    if not prompts:
        return 0.0
    ratios = []
    for value in values:
        best = 0.0
        for prompt in prompts:
            if not prompt:
                continue
            if prompt in value:
                best = max(best, 1.0)
            else:
                shared = len(set(prompt.split()) & set(value.split()))
                best = max(best, shared / max(1, len(set(prompt.split()))))
        ratios.append(best)
    return min(ratios) if ratios else 0.0


# --- streaming ------------------------------------------------------------


def infer_delta_path(events: Sequence[dict[str, Any]]) -> ExtractionResult:
    """Infer the delta path from stream *events*, not the reassembly (§8.5).

    Reassembling a stream requires already knowing the delta path, so walking
    the reassembled text to find it is circular. The event JSON is walked
    instead.
    """
    if not events:
        return ExtractionResult(
            path=None, confidence="none", method="delta_walk", evidence={"reason": "no events"}
        )

    counts: dict[str, int] = {}
    for event in events:
        for path, value in walk_string_paths(event):
            leaf = path.rsplit(".", 1)[-1].split("[")[0].lower()
            if leaf in STOPLIST or not value:
                continue
            counts[path] = counts.get(path, 0) + 1

    if not counts:
        return ExtractionResult(
            path=None,
            confidence="none",
            method="delta_walk",
            evidence={"reason": "no non-metadata string in any event"},
        )

    # The delta path is the one that appears in most events: metadata appears
    # once (in the opening frame), the delta appears in nearly all of them.
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    best_path, best_count = ranked[0]
    coverage = best_count / len(events)

    return ExtractionResult(
        path=best_path,
        confidence="high" if coverage >= 0.8 else "medium" if coverage >= 0.5 else "low",
        method="delta_walk",
        evidence={
            "events": len(events),
            "coverage": round(coverage, 2),
            "runner_up": ranked[1][0] if len(ranked) > 1 else None,
        },
    )
