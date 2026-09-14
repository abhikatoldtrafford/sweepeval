"""The judge itself (spec §11.9).

Three things this deliberately refuses to do.

**It will not be the target.** A judge whose resolved endpoint is the target's
is scoring its own output, and the number that comes out means nothing. That
is checked and refused outright. Model *family* is not knowable from a black
box -- a target behind a proxy may be anything -- so instead of pretending to
detect it, a shared vendor prefix between the judge model and any discovered
target model raises a warning, is recorded in the manifest, and proceeds. The
disclosure is the mitigation; a check that cannot actually work should not be
written as though it can.

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
from dataclasses import dataclass
from typing import Any

from sweepeval.judge.rubric import RUBRIC_VERSION, Rubric

__all__ = [
    "JudgeConfig",
    "JudgeError",
    "JudgeVerdict",
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


def refuse_if_same_endpoint(judge: JudgeConfig, target_url: str) -> None:
    """§11.9: the judge may not be the target.

    Compared on the resolved endpoint rather than the configured string, so
    the same host reached two ways still refuses.
    """
    from urllib.parse import urlsplit

    def fingerprint(url: str) -> tuple[str, str]:
        parts = urlsplit(url.strip().rstrip("/"))
        return (parts.netloc.lower(), parts.path.lower())

    if fingerprint(judge.url) == fingerprint(target_url):
        raise JudgeError(
            "the judge endpoint is the target endpoint. A model scoring its "
            "own output produces a number that means nothing (§11.9). Point "
            "--judge-url at a different endpoint, or drop --judge."
        )


def build_prompt(
    rubric: Rubric, *, probe: str, response: str, expectation: str
) -> str:
    return rubric.render(probe=probe, response=response, expectation=expectation)
