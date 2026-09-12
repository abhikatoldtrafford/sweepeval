"""Error classification and streaming decoders (spec §7)."""

from __future__ import annotations

import json

import httpx
import pytest

from sweepeval.http.errors import classify_exception, classify_status
from sweepeval.http.streaming import decode_chunked_json, decode_sse, reassemble
from sweepeval.schema.call import ErrorClass

# --- error table ----------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (200, ErrorClass.ok),
        (204, ErrorClass.ok),
        (400, ErrorClass.terminal),
        (401, ErrorClass.terminal),
        (403, ErrorClass.terminal),
        (404, ErrorClass.terminal),
        (422, ErrorClass.terminal),
        (408, ErrorClass.retryable),
        (429, ErrorClass.retryable),
        (500, ErrorClass.retryable),
        (502, ErrorClass.retryable),
        (503, ErrorClass.retryable),
        (504, ErrorClass.retryable),
    ],
)
def test_the_spec_table_maps_as_written(status: int, expected: ErrorClass) -> None:
    assert classify_status(status) is expected


def test_unknown_4xx_is_terminal() -> None:
    """Retrying a request the server rejected on its merits only wastes budget."""
    assert classify_status(418) is ErrorClass.terminal


def test_unknown_5xx_is_retryable() -> None:
    assert classify_status(599) is ErrorClass.retryable


def test_timeouts_and_network_errors_are_retryable() -> None:
    assert classify_exception(httpx.ReadTimeout("x")) is ErrorClass.retryable
    assert classify_exception(httpx.ConnectError("x")) is ErrorClass.retryable


def test_every_status_maps_to_exactly_one_class() -> None:
    for status in range(200, 600):
        assert isinstance(classify_status(status), ErrorClass)


# --- SSE ------------------------------------------------------------------

_SSE = (
    b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
    b'data: {"choices":[{"delta":{"content":"lo, "}}]}\n\n'
    b'data: {"choices":[{"delta":{"content":"world"}}]}\n\n'
    b"data: [DONE]\n\n"
)


def test_sse_reassembles_the_full_text() -> None:
    assert reassemble(decode_sse([_SSE])) == "Hello, world"


def test_sse_ignores_the_done_sentinel() -> None:
    assert "[DONE]" not in reassemble(decode_sse([_SSE]))


@pytest.mark.parametrize("split", range(1, len(_SSE)))
def test_sse_is_split_invariant_at_every_byte_boundary(split: int) -> None:
    """Network chunking is arbitrary.

    A decoder that assumed frame-aligned chunks would drop tokens
    non-deterministically, which is the worst possible failure mode for a
    determinism scorer — it would attribute harness noise to the target.
    """
    chunks = [_SSE[:split], _SSE[split:]]
    assert reassemble(decode_sse(chunks)) == "Hello, world"


def test_sse_handles_crlf_frames() -> None:
    payload = (
        b'data: {"choices":[{"delta":{"content":"a"}}]}\r\n\r\n'
        b'data: {"choices":[{"delta":{"content":"b"}}]}\r\n\r\n'
    )
    assert reassemble(decode_sse([payload])) == "ab"


def test_sse_ignores_comments_and_non_data_fields() -> None:
    payload = (
        b": keep-alive\n\n"
        b"event: message\n"
        b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
        b"id: 7\n\n"
    )
    assert reassemble(decode_sse([payload])) == "x"


def test_sse_tolerates_a_stream_ending_without_a_blank_line() -> None:
    payload = b'data: {"choices":[{"delta":{"content":"tail"}}]}'
    assert reassemble(decode_sse([payload])) == "tail"


def test_sse_passes_through_non_json_data_lines() -> None:
    assert reassemble(decode_sse([b"data: plain text\n\n"])) == "plain text"


def test_anthropic_style_deltas_reassemble() -> None:
    payload = (
        b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"He"}}\n\n'
        b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"y"}}\n\n'
    )
    assert reassemble(decode_sse([payload])) == "Hey"


def test_thinking_deltas_are_excluded_from_text() -> None:
    """§7: reasoning content never enters extracted text."""
    payload = (
        b'data: {"delta":{"type":"thinking_delta","thinking":"hmm let me think"}}\n\n'
        b'data: {"delta":{"type":"text_delta","text":"answer"}}\n\n'
    )
    assert reassemble(decode_sse([payload])) == "answer"


def test_sse_yields_the_raw_event_alongside_the_delta() -> None:
    events = list(decode_sse([b'data: {"choices":[{"delta":{"content":"x"}}],"id":"7"}\n\n']))
    assert events[0][0] == "x"
    assert events[0][1] is not None
    assert events[0][1]["id"] == "7"


# --- chunked JSON ---------------------------------------------------------

_JSONL = (
    b'{"text":"one "}\n'
    b'{"text":"two "}\n'
    b'{"text":"three"}\n'
)


def test_chunked_json_reassembles() -> None:
    assert reassemble(decode_chunked_json([_JSONL])) == "one two three"


@pytest.mark.parametrize("split", range(1, len(_JSONL)))
def test_chunked_json_is_split_invariant(split: int) -> None:
    chunks = [_JSONL[:split], _JSONL[split:]]
    assert reassemble(decode_chunked_json(chunks)) == "one two three"


def test_chunked_json_tolerates_a_missing_trailing_newline() -> None:
    assert reassemble(decode_chunked_json([b'{"text":"end"}'])) == "end"


def test_chunked_json_skips_unparseable_lines() -> None:
    payload = b'{"text":"a"}\nnot json\n{"text":"b"}\n'
    assert reassemble(decode_chunked_json([payload])) == "ab"


# --- equivalence ----------------------------------------------------------


def test_streamed_and_whole_forms_reassemble_identically() -> None:
    """§7: scoring must not be able to tell which transport was used."""
    text = "Hello, world"
    streamed = reassemble(decode_sse([_SSE]))
    whole = json.loads('{"choices":[{"message":{"content":"Hello, world"}}]}')[
        "choices"
    ][0]["message"]["content"]
    assert streamed == whole == text
