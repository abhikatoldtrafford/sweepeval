"""Error-guided mutation (spec §8.2).

A **finite, enumerated** mutation set, not a search. Error bodies are the
richest signal a black-box endpoint gives — a 400 usually names the field it
wanted — so the mutator reads the name out of the error and applies one of six
transformations along it.

Bounded on purpose: at most 2 mutations deep and 6 attempts total. "Guided
search" without a bound is how a tool ends up making hundreds of requests
against a stranger's endpoint.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MAX_MUTATION_ATTEMPTS",
    "MAX_MUTATION_DEPTH",
    "Mutation",
    "field_names_in_error",
    "mutations_for",
]

MAX_MUTATION_DEPTH = 2
MAX_MUTATION_ATTEMPTS = 6

_QUOTED = re.compile(r"['\"`]([A-Za-z_][A-Za-z0-9_.\-]{1,40})['\"`]")
_AFTER_KEYWORD = re.compile(
    r"(?:field|parameter|param|property|key|argument)\s+"
    r"['\"`]?([A-Za-z_][A-Za-z0-9_.\-]{1,40})['\"`]?",
    re.I,
)
_MISSING = re.compile(
    r"(?:missing|required|expected|unknown|unexpected|invalid)\s+"
    r"(?:required\s+)?(?:field|parameter|param|property|key)?\s*"
    r"['\"`]?([A-Za-z_][A-Za-z0-9_.\-]{1,40})['\"`]?",
    re.I,
)


@dataclass(frozen=True)
class Mutation:
    """One transformation, with the error text that motivated it."""

    kind: str
    field: str
    motivated_by: str

    def label(self) -> str:
        return f"{self.kind}:{self.field}"


def field_names_in_error(body: bytes | str, max_names: int = 4) -> list[str]:
    """Pull candidate field names out of an error body.

    Structured fields first — ``error.param`` is unambiguous where prose is a
    guess — then quoted names and keyword-adjacent names from the message.
    """
    text = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else body
    names: list[str] = []

    def add(name: str | None) -> None:
        if name and name not in names and name.isidentifier():
            names.append(name)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict):
            add(error.get("param"))
            add(error.get("field"))
            message = error.get("message")
            if isinstance(message, str):
                text = message
        add(parsed.get("param"))
        detail = parsed.get("detail")
        if isinstance(detail, list):
            for item in detail:
                if isinstance(item, dict):
                    loc = item.get("loc")
                    if isinstance(loc, list) and loc:
                        add(str(loc[-1]))

    for pattern in (_MISSING, _AFTER_KEYWORD, _QUOTED):
        for match in pattern.finditer(text):
            add(match.group(1))

    return names[:max_names]


def mutations_for(
    body: dict[str, Any], error_body: bytes | str, prompt: str = ""
) -> Iterator[tuple[Mutation, dict[str, Any]]]:
    """Yield ``(mutation, mutated_body)`` for each enumerated transformation.

    The six kinds of §8.2, in the order they are most likely to help.
    """
    names = field_names_in_error(error_body)
    if not names:
        return

    excerpt = (
        error_body.decode("utf-8", errors="replace")
        if isinstance(error_body, bytes)
        else error_body
    )[:120]

    payload_keys = [k for k in body if not k.startswith("__")]

    for name in names:
        if name in body:
            # 5. coerce a string to a single-element array, or the reverse
            value = body[name]
            if isinstance(value, str):
                yield (
                    Mutation("coerce_to_array", name, excerpt),
                    {**body, name: [value]},
                )
            elif isinstance(value, list) and len(value) == 1:
                yield (
                    Mutation("coerce_from_array", name, excerpt),
                    {**body, name: value[0]},
                )
            continue

        # 1. rename the offending field to the name the error mentions
        for key in payload_keys:
            renamed = {name if k == key else k: v for k, v in body.items()}
            yield (Mutation("rename", f"{key}->{name}", excerpt), renamed)
            break

        # 2. add a required field with a type-appropriate minimal value
        yield (
            Mutation("add", name, excerpt),
            {**body, name: _minimal_for(name, prompt)},
        )

        # 3. nest the payload under the named field
        yield (Mutation("nest", name, excerpt), {name: dict(body)})

    # 4. unwrap one level, when the body is a single-key wrapper
    if len(payload_keys) == 1:
        inner = body[payload_keys[0]]
        if isinstance(inner, dict):
            yield (Mutation("unwrap", payload_keys[0], excerpt), dict(inner))


def _minimal_for(name: str, prompt: str = "") -> Any:
    """A plausible value for a field we only know the name of.

    An unknown string field defaults to the **prompt**, not to ``""``. A field
    the endpoint demanded is very often the one carrying the prompt, and an
    empty string there produces the worst possible outcome: a 200 that conveys
    nothing. Discovery would then emit a config that looks healthy while every
    subsequent probe measures the target's response to an empty request.
    """
    lowered = name.lower()
    if any(token in lowered for token in ("token", "length", "count", "limit", "n_")):
        return 16
    if any(token in lowered for token in ("stream", "echo", "enable")):
        return False
    if any(token in lowered for token in ("messages", "contents", "inputs", "items")):
        return []
    if "model" in lowered:
        return "default"
    return prompt
