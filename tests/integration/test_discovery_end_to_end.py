"""Discovery end to end, per scenario (spec §8)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from tests.conftest import make_app, make_client

from sweepeval.discovery.emit import config_dir_for, emit_config, was_hand_edited
from sweepeval.discovery.runner import discover_target


async def _discover(name: str, key: str | None = None, url: str | None = None):
    app = make_app(name)
    client = make_client(app)
    try:
        target = url or ("https://mock.test" + app.scenario.paths[0])
        return await discover_target(client, target, key, seed=7)
    finally:
        await client.aclose()


# --- the headline case ----------------------------------------------------


async def test_the_oracle_beats_the_echo_end_to_end() -> None:
    """§8.5, D6. The defect rev 1 would have shipped.

    ``echoes_the_prompt`` mirrors the request in two fields that are longer
    than the answer, vary perfectly with input, and appear in every response.
    """
    outcome = await _discover("echoes_the_prompt")
    assert outcome.extraction.path == "$.choices[0].message.content"
    assert outcome.extraction.method == "nonce_oracle"
    assert outcome.extraction.confidence == "high"


async def test_the_oracle_finds_a_path_no_family_prior_covers() -> None:
    outcome = await _discover("weird_shape")
    assert outcome.extraction.path == "$.result.payload.answer.value"
    assert outcome.extraction.evidence["prior_agrees"] is False


async def test_extraction_agrees_with_the_prior_on_a_clean_target() -> None:
    outcome = await _discover("openai_clean", key="test-key-abcdefgh")
    assert outcome.extraction.path == "$.choices[0].message.content"
    assert outcome.extraction.evidence["prior_agrees"] is True


async def test_the_fallback_walk_runs_when_the_oracle_fails() -> None:
    """A target that refuses everything never emits the nonce."""
    outcome = await _discover("refuses_everything")
    assert outcome.extraction.method == "walk"
    assert outcome.extraction.evidence["oracle_failed"] is True


# --- target type ----------------------------------------------------------


async def test_a_plain_completion_endpoint_is_a_bare_model() -> None:
    outcome = await _discover("openai_clean", key="test-key-abcdefgh")
    assert outcome.target_type == "BARE_MODEL"


async def test_an_off_prior_envelope_is_an_agent_system() -> None:
    outcome = await _discover("weird_shape")
    assert outcome.target_type == "AGENT_SYSTEM"
    assert outcome.target_type_confidence == "low"


async def test_target_type_records_its_evidence_as_an_assumption() -> None:
    """§9: a wrong guess blocks a legitimate comparison, so it must be visible."""
    outcome = await _discover("weird_shape")
    assert set(outcome.target_type_evidence) == {
        "tool_structure",
        "retrieved_documents",
        "extraction_path_off_family_prior",
    }


# --- the emitted config ---------------------------------------------------


async def test_the_config_annotates_every_inference(tmp_path: Path) -> None:
    outcome = await _discover("openai_clean", key="test-key-abcdefgh")
    written = emit_config(tmp_path, "https://mock.test/v1/chat/completions",
                          outcome.payload)
    payload = yaml.safe_load(written.path.read_text(encoding="utf-8"))

    assert payload["target"]["shape"] == "openai.chat_completions"
    assert payload["target"]["shape_method"].startswith("stage B")
    assert payload["extraction"]["confidence"] == "high"
    assert payload["extraction"]["method"] == "nonce_oracle"
    assert payload["discovery"]["transcript"]


async def test_the_config_records_the_discovered_models(tmp_path: Path) -> None:
    outcome = await _discover("openai_clean", key="test-key-abcdefgh")
    assert outcome.payload["discovery"]["models_seen"] == [
        "gpt-mock-large",
        "gpt-mock-small",
    ]


async def test_a_query_param_key_is_redacted_from_the_config(tmp_path: Path) -> None:
    """§6.6: this file is written to disk and often committed."""
    outcome = await _discover("query_param_auth", key="sk-supersecretkeyvalue")
    written = emit_config(tmp_path, "https://mock.test/v1/chat/completions",
                          outcome.payload)
    assert "sk-supersecretkeyvalue" not in written.path.read_text(encoding="utf-8")


# --- reuse rules (D27) ----------------------------------------------------


def test_two_targets_in_one_directory_do_not_collide(tmp_path: Path) -> None:
    a = config_dir_for(tmp_path, "https://one.test/chat")
    b = config_dir_for(tmp_path, "https://two.test/chat")
    assert a != b


def test_a_freshly_written_config_is_not_seen_as_hand_edited(tmp_path: Path) -> None:
    written = emit_config(tmp_path, "https://x.test/chat", {"target": {"shape": "s"}})
    assert was_hand_edited(written.path) is False


def test_a_hand_edited_config_is_detected(tmp_path: Path) -> None:
    written = emit_config(tmp_path, "https://x.test/chat", {"target": {"shape": "s"}})
    payload = yaml.safe_load(written.path.read_text(encoding="utf-8"))
    payload["target"]["shape"] = "corrected.by.hand"
    written.path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    assert was_hand_edited(written.path) is True


def test_a_hand_edited_config_is_not_overwritten(tmp_path: Path) -> None:
    """D27: a corrected extraction path must never be silently thrown away."""
    written = emit_config(tmp_path, "https://x.test/chat", {"target": {"shape": "s"}})
    payload = yaml.safe_load(written.path.read_text(encoding="utf-8"))
    payload["target"]["shape"] = "corrected.by.hand"
    written.path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(FileExistsError, match="hand-edited"):
        emit_config(tmp_path, "https://x.test/chat", {"target": {"shape": "regenerated"}})


def test_force_overwrites_a_hand_edited_config(tmp_path: Path) -> None:
    written = emit_config(tmp_path, "https://x.test/chat", {"target": {"shape": "s"}})
    written.path.write_text("target: {shape: hand}\n", encoding="utf-8")
    again = emit_config(
        tmp_path, "https://x.test/chat", {"target": {"shape": "regenerated"}}, force=True
    )
    assert "regenerated" in again.path.read_text(encoding="utf-8")


def test_a_config_with_no_stamp_is_treated_as_hand_written(tmp_path: Path) -> None:
    path = tmp_path / "sweepeval.yaml"
    path.write_text("target: {shape: mine}\n", encoding="utf-8")
    assert was_hand_edited(path) is True


def test_the_config_header_tells_the_user_it_is_editable(tmp_path: Path) -> None:
    written = emit_config(tmp_path, "https://x.test/chat", {"target": {}})
    assert "meant to be edited" in written.path.read_text(encoding="utf-8")


# --- the discovered config must actually convey the prompt ----------------


@pytest.mark.parametrize(
    "scenario",
    ["openai_clean", "anthropic_streaming", "gemini_shape", "weird_shape",
     "echoes_the_prompt", "no_usage_block"],
)
async def test_the_discovered_request_actually_carries_the_prompt(scenario: str) -> None:
    """Guard against a 200 that conveys nothing.

    An `add:` mutation supplying an empty string produces a successful request
    that never sends the prompt. Discovery then emits a config that looks
    healthy while every later probe measures the target's response to an empty
    input — the metrics are real numbers about nothing.

    Proved by the oracle: if the target echoed a nonce we asked for, the
    request reached it.
    """
    keys = {
        "openai_clean": "test-key-abcdefgh",
        "anthropic_streaming": "test-key-abcdefgh",
        "gemini_shape": "test-key-abcdefgh",
    }
    outcome = await _discover(scenario, key=keys.get(scenario))
    assert outcome.extraction.method == "nonce_oracle", (
        f"{scenario}: the oracle failed, so the request may not be reaching "
        f"the target ({outcome.extraction.evidence})"
    )
    assert outcome.extraction.path is not None
