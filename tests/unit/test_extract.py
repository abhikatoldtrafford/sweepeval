"""Extraction inference (spec §8.5, D6).

The headline case is ``echoes_the_prompt``: the heuristic that rev 1 shipped
alone picks the echoed request over the real answer, and every text-derived
metric would then be computed on the tool's own prompt.
"""

from __future__ import annotations

from typing import Any

from sweepeval.discovery.extract import (
    STOPLIST,
    extract_at,
    infer_delta_path,
    infer_from_nonce,
    infer_from_walk,
    walk_string_paths,
)

OPENAI_PRIORS = ("$.choices[0].message.content", "$.choices[0].text")


def _openai(text: str, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "cmpl-1",
        "object": "chat.completion",
        "model": "m",
        "created": 0,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text},
             "finish_reason": "stop"}
        ],
    }
    body.update(extra)
    return body


# --- walking and resolution -----------------------------------------------


def test_walk_finds_nested_string_paths() -> None:
    paths = dict(walk_string_paths(_openai("hello")))
    assert paths["$.choices[0].message.content"] == "hello"
    assert paths["$.id"] == "cmpl-1"


def test_extract_at_round_trips_a_walked_path() -> None:
    body = _openai("hello")
    for path, value in walk_string_paths(body):
        assert extract_at(body, path) == value


def test_extract_at_joins_a_wildcard_in_document_order() -> None:
    """§8.5: $.content[*].text joins with no separator, in order."""
    body = {"content": [{"text": "He"}, {"text": "llo"}]}
    assert extract_at(body, "$.content[*].text") == "Hello"


def test_extract_at_skips_blocks_without_the_key() -> None:
    """Anthropic interleaves thinking blocks, which carry no `text`."""
    body = {"content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "ans"}]}
    assert extract_at(body, "$.content[*].text") == "ans"


def test_extract_at_returns_none_for_a_missing_path() -> None:
    assert extract_at(_openai("x"), "$.nope.nothing") is None


# --- the nonce oracle -----------------------------------------------------


def test_oracle_finds_the_path_carrying_the_nonce() -> None:
    result = infer_from_nonce(_openai("N0NCE7X2"), "N0NCE7X2")
    assert result.path == "$.choices[0].message.content"
    assert result.confidence == "high"
    assert result.method == "nonce_oracle"


def test_oracle_beats_the_echo() -> None:
    """The case the heuristic loses.

    Both the answer and the echoed request contain the nonce. The oracle
    prefers the deepest path and then the shortest value, so it takes the
    answer and not the instruction that asked for it.
    """
    body = _openai(
        "N0NCE7X2",
        echo="Reply with exactly the following and nothing else: N0NCE7X2",
    )
    assert infer_from_nonce(body, "N0NCE7X2").path == "$.choices[0].message.content"


def test_oracle_finds_a_buried_answer_no_prior_would_reach() -> None:
    """The weird_shape case: $.result.payload.answer.value."""
    body = {"result": {"payload": {"answer": {"value": "N0NCE7X2"}}}, "ok": True}
    assert infer_from_nonce(body, "N0NCE7X2").path == "$.result.payload.answer.value"


def test_oracle_reports_failure_when_the_nonce_is_absent() -> None:
    result = infer_from_nonce(_openai("I won't do that"), "N0NCE7X2")
    assert result.path is None
    assert result.ok is False
    assert "no value contained the nonce" in result.evidence["reason"]


def test_oracle_records_other_matches_for_the_report() -> None:
    body = _openai("N0NCE7X2", echo="say N0NCE7X2")
    result = infer_from_nonce(body, "N0NCE7X2")
    assert result.evidence["unique_match"] is False
    assert result.evidence["other_matches"]


# --- the fallback walk ----------------------------------------------------


def test_walk_finds_the_answer_on_a_clean_target() -> None:
    payloads = [_openai("a long and distinctive first answer"),
                _openai("a completely different second answer")]
    result = infer_from_walk(payloads, prompts=["q1", "q2"], priors=OPENAI_PRIORS)
    assert result.path == "$.choices[0].message.content"


def test_walk_rejects_the_echoed_prompt() -> None:
    """The rev-1 defect, pinned.

    The echo is longer than the answer, varies perfectly with input, and is
    present in every probe — it wins on every positive term. Only the
    near-duplicate penalty stops it.
    """
    prompts = [
        "a distinctive and rather long first question about widgets",
        "an entirely different and equally long second question about gadgets",
    ]
    payloads = [_openai("OK", echo=p, request_text=p) for p in prompts]
    result = infer_from_walk(payloads, prompts=prompts, priors=OPENAI_PRIORS)
    assert result.path == "$.choices[0].message.content"


def test_walk_penalises_the_echo_even_without_the_stoplist() -> None:
    """A framework's mirror field can be called anything."""
    prompts = ["a distinctive and rather long first question about widgets",
               "an entirely different and equally long second question"]
    payloads = [_openai("OK", mirrored_user_turn=p) for p in prompts]
    result = infer_from_walk(payloads, prompts=prompts, priors=OPENAI_PRIORS)
    assert result.path == "$.choices[0].message.content"


def test_walk_penalises_verbose_error_envelopes() -> None:
    payloads = [
        {"error": {"message": "a very long and explanatory error about q1 " * 3},
         "text": "short"},
        {"error": {"message": "a very long and explanatory error about q2 " * 3},
         "text": "brief"},
    ]
    result = infer_from_walk(payloads, prompts=["q1", "q2"])
    assert result.path == "$.text"


def test_walk_prefers_paths_present_in_every_probe() -> None:
    payloads = [_openai("first"), _openai("second", occasional="only here")]
    result = infer_from_walk(payloads, prompts=["a", "b"], priors=OPENAI_PRIORS)
    assert result.path == "$.choices[0].message.content"


def test_walk_confidence_falls_when_the_margin_is_thin() -> None:
    payloads = [{"a": "one answer here", "b": "other answer here"},
                {"a": "one answer there", "b": "other answer there"}]
    assert infer_from_walk(payloads, prompts=["x", "y"]).confidence in {"low", "medium"}


def test_walk_reports_no_candidates_on_an_empty_body() -> None:
    assert infer_from_walk([{}], prompts=["x"]).path is None


def test_the_stoplist_covers_the_request_mirror_names() -> None:
    for name in ("prompt", "input", "echo", "request", "request_text", "messages"):
        assert name in STOPLIST


# --- streaming ------------------------------------------------------------


def test_delta_path_is_inferred_from_events_not_the_reassembly() -> None:
    """Reassembly needs the delta path, so walking it would be circular."""
    events = [
        {"id": "1", "object": "chunk", "choices": [{"delta": {"content": "He"}}]},
        {"id": "1", "object": "chunk", "choices": [{"delta": {"content": "llo"}}]},
        {"id": "1", "object": "chunk", "choices": [{"delta": {"content": "!"}}]},
    ]
    result = infer_delta_path(events)
    assert result.path == "$.choices[0].delta.content"
    assert result.confidence == "high"


def test_delta_inference_ignores_metadata_that_appears_once() -> None:
    events = [
        {"type": "message_start", "message": {"role": "assistant"}},
        {"delta": {"text": "a"}},
        {"delta": {"text": "b"}},
        {"delta": {"text": "c"}},
    ]
    assert infer_delta_path(events).path == "$.delta.text"


def test_delta_inference_reports_failure_on_empty_events() -> None:
    assert infer_delta_path([]).path is None
