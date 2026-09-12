"""Streaming decoders (spec §7).

SSE and chunked-JSON sit behind one iterator yielding ``(delta_text,
raw_event)``; the non-streaming path yields a single element. Nothing
downstream knows which it got, so scoring is transport-agnostic and the
reassembled text is identical either way.

The decoders are byte-oriented and buffer across chunk boundaries, because
network chunking is arbitrary: an SSE frame can and does arrive split across
two reads, and a decoder that assumed frame-aligned chunks would drop tokens
non-deterministically — the worst possible failure for a determinism scorer.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any

__all__ = ["Event", "decode_chunked_json", "decode_sse", "reassemble"]

Event = tuple[str, dict[str, Any] | None]
"""``(delta_text, raw_event)``. ``raw_event`` is None for opaque payloads."""

_DONE_SENTINELS = frozenset({"[DONE]", "DONE"})


def decode_sse(chunks: Iterable[bytes]) -> Iterator[Event]:
    """Decode a Server-Sent Events stream.

    Frames are separated by a blank line. Only ``data:`` lines carry payload;
    ``event:``, ``id:``, ``retry:`` and comments are structure we surface but do
    not treat as text.
    """
    buffer = b""
    for chunk in chunks:
        buffer += chunk
        while True:
            frame, sep, rest = _split_frame(buffer)
            if not sep:
                break
            buffer = rest
            event = _parse_sse_frame(frame)
            if event is not None:
                yield event

    # A stream that ends without a trailing blank line still has a final frame.
    if buffer.strip():
        event = _parse_sse_frame(buffer)
        if event is not None:
            yield event


def _split_frame(buffer: bytes) -> tuple[bytes, bool, bytes]:
    for sep in (b"\n\n", b"\r\n\r\n"):
        head, found, tail = buffer.partition(sep)
        if found:
            return head, True, tail
    return buffer, False, b""


def _parse_sse_frame(frame: bytes) -> Event | None:
    data_lines: list[str] = []
    for raw_line in frame.split(b"\n"):
        line = raw_line.decode("utf-8", errors="replace").strip("\r")
        if not line or line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if field == "data":
            data_lines.append(value.lstrip())

    if not data_lines:
        return None

    payload = "\n".join(data_lines)
    if payload.strip() in _DONE_SENTINELS:
        return None

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        # Some targets stream raw text in data: lines rather than JSON.
        return (payload, None)

    if isinstance(parsed, dict):
        return (_delta_text(parsed), parsed)
    return ("", None)


def decode_chunked_json(chunks: Iterable[bytes]) -> Iterator[Event]:
    """Decode newline-delimited JSON (JSON Lines) streaming."""
    buffer = b""
    for chunk in chunks:
        buffer += chunk
        while b"\n" in buffer:
            line, _, buffer = buffer.partition(b"\n")
            event = _parse_json_line(line)
            if event is not None:
                yield event

    if buffer.strip():
        event = _parse_json_line(buffer)
        if event is not None:
            yield event


def _parse_json_line(line: bytes) -> Event | None:
    text = line.decode("utf-8", errors="replace").strip()
    if not text or text in _DONE_SENTINELS:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return (_delta_text(parsed), parsed)
    return ("", None)


# Known delta locations, tried in order. Discovery infers the real path for an
# unknown shape (§8.5); these cover the shapes the ladder already identifies,
# so streaming works during discovery itself.
def _delta_text(event: dict[str, Any]) -> str:
    choices = event.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            delta = first.get("delta")
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str):
                    return content
            text = first.get("text")
            if isinstance(text, str):
                return text

    # Anthropic content_block_delta
    delta = event.get("delta")
    if isinstance(delta, dict):
        # thinking deltas are reasoning, not answer text (§7)
        if delta.get("type") == "thinking_delta":
            return ""
        text = delta.get("text")
        if isinstance(text, str):
            return text

    for key in ("text", "content", "output_text", "response"):
        value = event.get(key)
        if isinstance(value, str):
            return value

    return ""


def reassemble(events: Iterable[Event]) -> str:
    """Join deltas in arrival order into the full response text."""
    return "".join(delta for delta, _ in events)
