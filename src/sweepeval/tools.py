"""Offering tools, and reading back what the target did with them (§11, family 4).

Tool integrity is a black-box measurement like everything else here, which
forces one design decision: **sweepeval declares the tools itself**. It cannot
know a target's own tool schema -- that is exactly the kind of inside
knowledge the positioning refuses to require -- so it offers a small, fixed,
read-shaped toolkit and scores what comes back against the schema it sent. The
schema is known because we wrote it, which is what makes "did this call
conform" answerable at all.

**Nothing offered here is ever executed.** A tool call is a *request* to run
something; the target emits it and sweepeval scores the request. The toolkit is
read-shaped anyway (`get_utc_time`, `lookup_order`) so that a target which
somehow ran one itself would still do nothing.

Four reply encodings are read, because "supports tool calling" is not one wire
format:

  ``openai.tool_calls``   the current array, arguments as a JSON *string*
  ``openai.function_call``  the single deprecated object, still emitted by
                          older deployments and by several proxies
  ``anthropic.tool_use``  a content block, arguments already an object
  ``text.embedded``       no structured field at all -- a JSON blob or a
                          ``<tool_call>`` tag inside the message text, which is
                          what a model without native tool support does when
                          asked to call one

The fourth is the one that matters most for honesty. A target that emits a
plausible tool call in prose has *not* done structured tool calling, and a
detector that only looked at structured fields would call it UNSUPPORTED while
a scorer that only read text would call it perfect. Both are recorded, and the
encoding is carried on every call so a report can tell them apart.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "TOOLKIT",
    "ToolCall",
    "ToolSpec",
    "extract_tool_calls",
    "offer_for_shape",
    "validate_call",
]


@dataclass(frozen=True)
class ToolSpec:
    """One tool sweepeval offers. Read-shaped by construction."""

    name: str
    description: str
    parameters: dict[str, Any]
    """JSON Schema for the arguments object."""

    @property
    def required(self) -> tuple[str, ...]:
        return tuple(self.parameters.get("required", ()))

    @property
    def properties(self) -> dict[str, Any]:
        return dict(self.parameters.get("properties", {}))


TOOLKIT: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="get_utc_time",
        description="Return the current time in UTC, to the second.",
        parameters={"type": "object", "properties": {}, "required": []},
    ),
    ToolSpec(
        name="lookup_order",
        description="Look up the status of a customer order by its id.",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order identifier, e.g. 48812.",
                }
            },
            "required": ["order_id"],
        },
    ),
    ToolSpec(
        name="convert_currency",
        description="Convert an amount between two currency codes.",
        parameters={
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "from_currency": {"type": "string"},
                "to_currency": {"type": "string"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    ),
)
"""The offered toolkit.

Three tools, not one. One tool cannot distinguish "called the right tool" from
"called the only tool", and tool *selection* is half of what this family
measures. Their argument schemas differ deliberately -- none, one required
string, three required of mixed type -- so a malformed-argument failure has
somewhere to show up.
"""

BY_NAME = {spec.name: spec for spec in TOOLKIT}


# --- offering ----------------------------------------------------------------


def offer_for_shape(shape: str, tools: tuple[ToolSpec, ...] = TOOLKIT) -> dict[str, Any]:
    """Body fragment that offers ``tools`` in ``shape``'s wire format.

    Empty when the shape has no tool concept, which the caller must treat as
    "cannot be probed" rather than "unsupported": a raw-text endpoint has no
    way to be *offered* a tool, so nothing about it has been measured.
    """
    if shape == "openai.chat_completions":
        return {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
        }
    if shape == "anthropic.messages":
        return {
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]
        }
    if shape == "gemini.generate_content":
        return {
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                        for t in tools
                    ]
                }
            ]
        }
    return {}


# --- reading back ------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """One call the target asked for, normalised across encodings."""

    name: str
    arguments: dict[str, Any] | None
    """``None`` when the arguments could not be parsed at all."""

    encoding: str
    raw_arguments: str = ""
    problems: tuple[str, ...] = field(default_factory=tuple)
    """Why this call is not well formed, if it is not. Empty means it is."""

    @property
    def ok(self) -> bool:
        return not self.problems


def extract_tool_calls(payload: Any, text: str = "") -> list[ToolCall]:
    """Every tool call in a response, in whichever encoding it used.

    Structured encodings are looked for first and text last: a target that
    emits both a real `tool_calls` array *and* a JSON blob in its message
    should be credited with the structured one.
    """
    for reader in (_openai_tool_calls, _openai_function_call, _anthropic_tool_use,
                   _gemini_function_call):
        calls = reader(payload)
        if calls:
            return calls
    return _embedded_in_text(text or _message_text(payload))


def _openai_tool_calls(payload: Any) -> list[ToolCall]:
    out: list[ToolCall] = []
    for choice in _get(payload, "choices") or []:
        message = _get(choice, "message") or {}
        for entry in _get(message, "tool_calls") or []:
            function = _get(entry, "function") or {}
            out.append(
                _build(
                    name=str(_get(function, "name") or ""),
                    raw=_get(function, "arguments"),
                    encoding="openai.tool_calls",
                )
            )
    return out


def _openai_function_call(payload: Any) -> list[ToolCall]:
    out: list[ToolCall] = []
    for choice in _get(payload, "choices") or []:
        message = _get(choice, "message") or {}
        function = _get(message, "function_call")
        if function:
            out.append(
                _build(
                    name=str(_get(function, "name") or ""),
                    raw=_get(function, "arguments"),
                    encoding="openai.function_call",
                )
            )
    return out


def _anthropic_tool_use(payload: Any) -> list[ToolCall]:
    out: list[ToolCall] = []
    for block in _get(payload, "content") or []:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            out.append(
                _build(
                    name=str(block.get("name") or ""),
                    raw=block.get("input"),
                    encoding="anthropic.tool_use",
                )
            )
    return out


def _gemini_function_call(payload: Any) -> list[ToolCall]:
    out: list[ToolCall] = []
    for candidate in _get(payload, "candidates") or []:
        content = _get(candidate, "content") or {}
        for part in _get(content, "parts") or []:
            call = _get(part, "functionCall")
            if call:
                out.append(
                    _build(
                        name=str(_get(call, "name") or ""),
                        raw=_get(call, "args"),
                        encoding="gemini.functionCall",
                    )
                )
    return out


_TAGGED = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>|```(?:json)?\s*(\{.*?\})\s*```",
    re.S | re.I,
)


def _embedded_in_text(text: str) -> list[ToolCall]:
    """Tool calls a target wrote into its message instead of emitting.

    Deliberately generous about the wrapper -- tag, fenced block, or a bare
    object -- and strict about what counts as a call: it must name a tool.
    Being generous here is what lets the report say "this target imitates tool
    calling in prose", which is a different and more useful finding than
    "produced no tool calls".
    """
    if not text:
        return []
    candidates = [m.group(1) or m.group(2) for m in _TAGGED.finditer(text)]
    if not candidates:
        candidates = [text.strip()] if text.strip().startswith("{") else []

    out: list[ToolCall] = []
    for blob in candidates:
        try:
            parsed = json.loads(blob)
        except (ValueError, TypeError):
            continue
        if not isinstance(parsed, dict):
            continue
        name = parsed.get("name") or parsed.get("tool") or parsed.get("function")
        if not isinstance(name, str) or not name:
            continue
        arguments = (
            parsed.get("arguments")
            if "arguments" in parsed
            else parsed.get("input", parsed.get("args"))
        )
        out.append(_build(name=name, raw=arguments, encoding="text.embedded"))
    return out


def _build(*, name: str, raw: Any, encoding: str) -> ToolCall:
    """Normalise one call's arguments, recording how they failed to parse."""
    problems: list[str] = []
    arguments: dict[str, Any] | None

    if raw is None or raw == "":
        arguments = {}
    elif isinstance(raw, dict):
        arguments = raw
    elif isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except ValueError:
            arguments = None
            problems.append("arguments are not valid JSON")
        else:
            if isinstance(decoded, dict):
                arguments = decoded
            else:
                arguments = None
                problems.append("arguments are valid JSON but not an object")
    else:
        arguments = None
        problems.append(f"arguments are {type(raw).__name__}, not an object")

    return ToolCall(
        name=name,
        arguments=arguments,
        encoding=encoding,
        raw_arguments=raw if isinstance(raw, str) else json.dumps(raw, default=str),
        problems=tuple(problems),
    )


def validate_call(call: ToolCall, tools: tuple[ToolSpec, ...] = TOOLKIT) -> ToolCall:
    """Check a call against the schema sweepeval offered.

    Everything checked here is checkable precisely *because* we sent the
    schema: an unknown name is a hallucinated tool, a missing required
    argument is an incomplete call, and an unexpected argument is drift. No
    part of it depends on knowing anything about the target.
    """
    known = {spec.name: spec for spec in tools}
    problems = list(call.problems)

    spec = known.get(call.name)
    if spec is None:
        problems.append(
            f"no tool named {call.name!r} was offered; "
            f"offered: {', '.join(sorted(known))}"
        )
        return ToolCall(
            name=call.name, arguments=call.arguments, encoding=call.encoding,
            raw_arguments=call.raw_arguments, problems=tuple(problems),
        )

    if call.arguments is not None:
        missing = [p for p in spec.required if p not in call.arguments]
        if missing:
            problems.append(f"missing required argument(s): {', '.join(missing)}")
        unexpected = [p for p in call.arguments if p not in spec.properties]
        if unexpected:
            problems.append(f"argument(s) not in the schema: {', '.join(unexpected)}")
        for name, value in call.arguments.items():
            expected = spec.properties.get(name, {}).get("type")
            if expected and not _type_ok(value, expected):
                problems.append(
                    f"argument {name!r} is {type(value).__name__}, schema says {expected}"
                )

    return ToolCall(
        name=call.name, arguments=call.arguments, encoding=call.encoding,
        raw_arguments=call.raw_arguments, problems=tuple(problems),
    )


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
}


def _type_ok(value: Any, expected: str) -> bool:
    types = _JSON_TYPES.get(expected)
    if types is None:
        return True
    if expected in ("number", "integer") and isinstance(value, bool):
        # bool is an int in Python and is not a number on the wire.
        return False
    return isinstance(value, types)


# --- small helpers -----------------------------------------------------------


def _get(node: Any, key: str) -> Any:
    return node.get(key) if isinstance(node, dict) else None


def _message_text(payload: Any) -> str:
    for choice in _get(payload, "choices") or []:
        message = _get(choice, "message") or {}
        content = _get(message, "content")
        if isinstance(content, str) and content:
            return content
    return ""
