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

    require_fields: tuple[str, ...] = ()
    """Fields the endpoint 400s without, announced the way OpenAI announces it.

    Every other scenario answers a bare ``{"messages": [...]}``, so nothing in
    CI ever reached §8.2's error-guided mutation: the ladder found a 200 on its
    first shape and stopped. That left the whole mutation path -- the part of
    discovery that talks to an endpoint which has already said no -- covered
    only by unit tests over canned error strings.

    It is not a hypothetical gap. Pointing discovery at api.openai.com on
    2026-09-12 found three bugs living exactly here, all of them invisible to
    the suite. The wording below is deliberately OpenAI's: prose, unquoted,
    ``error.param`` null, and the field name *before* the keyword.

    When ``model`` is required and ``expose_models`` is non-empty, the value is
    checked against that list too, so a mutation that invents a plausible
    string rather than reading one off ``/v1/models`` still fails. That was the
    third bug: the mutator guessed ``model="default"``, which no provider
    serves.
    """

    # --- behaviour --------------------------------------------------------
    echoes_prompt: bool = False
    """Mirror the request text back in a response field.

    The single most important knob for testing extraction: an echoed prompt
    outscores the real answer on every heuristic the blind walk uses (§8.5).
    """

    temperature_effect: bool = True
    nondeterministic_at_temp0: bool = False
    cache_responses: bool = False

    reads_its_input: bool = False
    """Answer a single-turn question by quoting that turn's own earlier text.

    The degradation family plants a fact, buries it under filler and asks for
    it back in one turn. `_recall` only fires from turn two onward -- it was
    written for the context family -- so without this the mock cannot recall
    within a turn and every degradation probe fails, which would make "the
    scorer works" indistinguishable from "the scorer always fails".

    A knob rather than a default, because the obvious generalisation is
    dangerous: security probes are single-turn and carry a canary, and a mock
    that quoted any single turn back would manufacture a leak on every one of
    them. Canary-shaped tokens are stripped here as well, for the same reason
    they are in `_recall`.
    """

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
    throttle_above_concurrency: int | None = None
    """429 any request that arrives while more than N others are in flight.

    A rate limiter that trips on concurrency rather than on a running total,
    which is what real ones mostly do -- and the only fixture that can throttle
    the degradation ramp *without* also throttling its serial control. A
    counter-based `rate_limit_after` cannot: by the time the burst runs, the
    control has already spent the same budget, both sides fail, and the paired
    verdict correctly reports that nothing is attributable to load.
    """

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
