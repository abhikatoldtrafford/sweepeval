"""Scenario definition for the mock endpoint (spec §17).

A scenario is data. Everything the mock does — which shape it speaks, which
probes it fails, whether temperature has any effect, how deep it remembers —
is declared here, so a test that needs a leaky, non-deterministic, caching
target does not need a line of Python.

This is what makes discovery testable without spending tokens, and it is why
§17 puts the mock in M1 rather than in a testing milestone at the end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["Scenario", "ShapeName", "load_scenario", "load_scenario_dir"]

ShapeName = Literal[
    "openai",
    "anthropic",
    "gemini",
    "prompt",
    "input",
    "raw",
    "weird",
]

AuthStyle = Literal["bearer", "x-api-key", "api-key", "query", "none"]


class Scenario(BaseModel):
    """Everything the mock's behaviour is derived from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str = ""

    # --- shape and transport ---------------------------------------------
    shape: ShapeName = "openai"
    paths: tuple[str, ...] = ("/v1/chat/completions",)
    """Paths that answer. Anything else 404s, which drives stage C."""

    supports_streaming: bool = True
    emit_usage: bool = True
    emit_usage_when_streaming: bool = True
    expose_models: tuple[str, ...] = ()
    """Served at /v1/models. Empty means the endpoint has no model list."""

    expose_openapi: bool = False
    auth: AuthStyle = "bearer"
    expected_key: str | None = None

    # --- behaviour --------------------------------------------------------
    echoes_prompt: bool = False
    """Mirror the request text back in a response field.

    The single most important knob for testing extraction: an echoed prompt
    outscores the real answer on every heuristic the blind walk uses (§8.5).
    """

    temperature_effect: bool = True
    nondeterministic_at_temp0: bool = False
    cache_responses: bool = False

    supports_system_prompt: bool = True
    supports_multi_turn: bool = True
    context_drop_depth: int | None = None
    """Turns beyond this depth are forgotten."""

    emits_reasoning: bool = False
    leaks_guardrails: bool = False
    quotes_the_canary: bool = False
    """Refuse, but quote the attack text — the hard-fail false positive (§11.2)."""

    hedges_guardrails: bool = False
    """Answer guardrail probes in general terms without disclosing anything.

    The band no lexical rule can classify, and the reason §11.9 specifies a
    judge. Modelled on live gpt-6-astra responses: neither the scripted
    refusal nor the scripted leak, which is what every other scenario
    produces and why the judge had nothing to escalate in any test.
    """

    leaks_system_prompt: bool = False
    refuses_everything: bool = False
    refusal_text: str = "I can't help with that request."

    # --- faults -----------------------------------------------------------
    fail_paths: tuple[str, ...] = ()
    malformed_json: bool = False
    latency_ms: float = 0.0
    error_rate: float = 0.0
    rate_limit_after: int | None = None
    retry_after_s: float | None = None
    max_input_chars: int | None = None

    extra: dict[str, Any] = Field(default_factory=dict)


def load_scenario(path: Path | str) -> Scenario:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Scenario.model_validate(payload)


def load_scenario_dir(directory: Path | str) -> dict[str, Scenario]:
    scenarios: dict[str, Scenario] = {}
    for path in sorted(Path(directory).glob("*.yaml")):
        scenario = load_scenario(path)
        scenarios[scenario.name] = scenario
    return scenarios
