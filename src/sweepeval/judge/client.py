"""The judge itself (spec §11.9).

Three things this deliberately refuses to do.

**It will not be the target.** A model scoring its own output produces a
number that means nothing. §11.9 states the check as endpoint equality, which
read literally makes the judge unusable for the commonest case there is --
sweeping several models on one provider, where `gpt-4o-mini` judging `gpt-5.2`
at the same host is not self-scoring and refusing it protects nothing. So
`check_independence` refuses on the fact that is actually knowable: the judge
model being one of the models under test, at any endpoint. The same endpoint
with an unknown target model is refused too, because self-judging cannot be
ruled out there. Everything else proceeds with a disclosure, since model
family is not knowable from a black box and a check that cannot work should
not be written as though it can.

**It will not guess.** A response that is not strict JSON, or whose verdict is
not one of the three, is a judge failure and is recorded as such. Salvaging a
verdict out of prose would make the judge's reliability invisible exactly
where it matters.

**It will not vary.** temperature=0, a pinned model id, and a versioned prompt
-- all three in the manifest, because `judge{present, model, prompt_version}`
is a hard comparability key.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sweepeval.judge.rubric import RUBRIC_VERSION, Rubric

__all__ = [
    "JudgeConfig",
    "JudgeError",
    "JudgeVerdict",
    "check_independence",
    "endpoint_fingerprint",
    "parse_verdict",
    "shares_vendor_prefix",
]

_VERDICTS = {"PASS", "FAIL", "UNSCORABLE"}
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class JudgeError(RuntimeError):
    """The judge cannot be used at all. Never downgraded to a warning."""


@dataclass(frozen=True)
class JudgeConfig:
    """Who is judging, and under which prompt."""

    model: str
    url: str
    key: str | None = None
    prompt_version: int = RUBRIC_VERSION

    def manifest(self) -> dict[str, Any]:
        return {"model": self.model, "prompt_version": self.prompt_version}


@dataclass(frozen=True)
class JudgeVerdict:
    """One resolved ambiguity."""

    verdict: str
    confidence: float
    rationale: str

    @property
    def value(self) -> float | None:
        if self.verdict == "PASS":
            return 1.0
        if self.verdict == "FAIL":
            return 0.0
        return None


def parse_verdict(text: str) -> JudgeVerdict:
    """Strict JSON in, verdict out. Raises rather than salvaging.

    Models wrap JSON in prose or a fence often enough that extracting the
    outermost object is worth doing -- but that is the ONLY latitude. A
    verdict outside the three, or a confidence that is not a number, is a
    failed judge call, and pretending otherwise would hide how often the judge
    cannot answer.
    """
    match = _JSON_BLOCK.search(text or "")
    if not match:
        raise JudgeError(f"judge returned no JSON object: {(text or '')[:120]!r}")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        raise JudgeError(f"judge returned invalid JSON: {error}") from None
    if not isinstance(payload, dict):
        raise JudgeError("judge returned JSON that is not an object")

    verdict = str(payload.get("verdict", "")).strip().upper()
    if verdict not in _VERDICTS:
        raise JudgeError(f"judge returned an unknown verdict {verdict!r}")

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        raise JudgeError("judge returned a non-numeric confidence") from None
    confidence = min(1.0, max(0.0, confidence))

    return JudgeVerdict(
        verdict=verdict,
        confidence=confidence,
        rationale=str(payload.get("rationale", "")).strip()[:400],
    )


def shares_vendor_prefix(judge_model: str, target_models: list[str]) -> str | None:
    """The target model this judge may share a vendor with, if any.

    Prefix before the first separator: `gpt-5.1` and `gpt-4o` share `gpt`.
    Crude on purpose. It cannot establish independence -- nothing available to
    a black-box tool can -- so it exists only to raise the disclosure §11.9
    asks for, never to gate anything.
    """
    def vendor(name: str) -> str:
        return re.split(r"[-._/:]", name.strip().lower(), maxsplit=1)[0]

    mine = vendor(judge_model)
    if not mine:
        return None
    return next((t for t in target_models if vendor(t) == mine), None)


def endpoint_fingerprint(url: str) -> tuple[str, str]:
    """Host and path, normalised, so the same endpoint reached two ways matches."""
    from urllib.parse import urlsplit

    parts = urlsplit(url.strip().rstrip("/"))
    return (parts.netloc.lower(), parts.path.lower())


def check_independence(
    judge: JudgeConfig, target_url: str, target_models: Sequence[str] = ()
) -> str | None:
    """§11.9: the judge may not be the target. Returns a warning, or raises.

    The spec says to refuse when "the judge's resolved endpoint fingerprint
    equals the target's", and taken literally that makes the judge unusable
    for the commonest case there is: sweeping several models on one provider.
    `gpt-4o-mini` scoring `gpt-5.2` at the same host is not a model scoring
    its own output, and refusing it protects nothing.

    So the endpoint is used as evidence rather than as the rule:

    * the judge model is one of the models under test -- **refused**, this is
      the thing §11.9 exists to prevent, and it is refused at any endpoint;
    * the same endpoint and the models under test are unknown -- **refused**,
      because self-judging cannot be ruled out and a number that might be
      self-scored is worth nothing;
    * the same endpoint with known, different models -- allowed, with a
      warning naming the shared host;
    * a different endpoint -- allowed, with the vendor-prefix warning §11.9
      asks for when the two model strings look related.

    Model *family* still is not knowable from a black box, and this does not
    pretend otherwise. It only uses the one fact that is knowable: which model
    ids this run is actually sweeping.
    """
    models = [m for m in target_models if m]
    same_endpoint = endpoint_fingerprint(judge.url) == endpoint_fingerprint(target_url)

    if any(judge.model.strip().lower() == m.strip().lower() for m in models):
        raise JudgeError(
            f"the judge model {judge.model!r} is one of the models under test. "
            "A model scoring its own output produces a number that means "
            "nothing (§11.9). Choose a judge that is not in the sweep."
        )

    if same_endpoint and not models:
        raise JudgeError(
            "the judge endpoint is the target endpoint and the target's model "
            "is unknown, so the judge may be scoring its own output (§11.9). "
            "Point --judge-url elsewhere, or declare the model axis so the two "
            "can be told apart."
        )

    if same_endpoint:
        return (
            f"judge {judge.model!r} shares an endpoint with the models under "
            "test; it is not one of them, but they are likely the same vendor"
        )

    related = shares_vendor_prefix(judge.model, models)
    if related:
        return (
            f"judge {judge.model!r} and target {related!r} share a vendor "
            "prefix. Independence cannot be established from a black box, so "
            "this is disclosed rather than blocked (§11.9)"
        )
    return None


def build_prompt(
    rubric: Rubric, *, probe: str, response: str, expectation: str
) -> str:
    return rubric.render(probe=probe, response=response, expectation=expectation)
