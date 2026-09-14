"""Spending the judge calls (spec §11.9).

Every call is a row in ``calls.jsonl`` tagged ``role: judge`` with its response
in the blob store, so §5.1's offline rebuild and I7 both hold: a reader can
reconstruct which ambiguity produced which verdict without the network, and
without trusting this module's summary of it.

A judge that fails is not a judge that passed. Any call that errors, or comes
back as something other than strict JSON, leaves the deterministic UNSCORABLE
in place and is counted in :attr:`JudgeOutcome.failures`. Falling back to a
guess would put a fabricated number on the frontier under a label that says a
model decided it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sweepeval.discovery.extract import extract_at
from sweepeval.http.client import CallResult, TransportClient
from sweepeval.judge.client import (
    JudgeConfig,
    JudgeError,
    build_prompt,
    parse_verdict,
)
from sweepeval.judge.escalate import Escalation, resolved_observation
from sweepeval.judge.rubric import rubric_for
from sweepeval.schema.observation import Observation

__all__ = ["JUDGE_TEXT_PATH", "JudgeOutcome", "aresolve"]

JUDGE_TEXT_PATH = "$.choices[0].message.content"
"""Where the judge's reply lives.

Fixed, not discovered. The judge is an endpoint *we* configure and prompt, so
running the six-shape discovery ladder against it would spend requests to
learn something the flag already stated. `TransportClient` leaves `text` empty
for a non-streaming response -- extraction is the caller's job -- so this is
that job.

In `extract_at`'s own path syntax, not a dotted one: `choices.0.message.content`
silently resolves to nothing, and "nothing" reads as a judge that returned no
JSON.
"""


@dataclass
class JudgeOutcome:
    """What the judge decided, and what it could not."""

    observations: list[Observation] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)
    """``(unit_id, reason)`` per call that errored or returned unusable JSON.
    Surfaced in the report: a judge that silently fails half its calls looks
    exactly like a judge that resolved them."""

    requests: int = 0
    tokens_out: int = 0

    calls: list[CallResult] = field(default_factory=list)
    """Every judge attempt, for the caller to persist. §11.9 requires a row in
    ``calls.jsonl`` tagged ``role: judge`` so the offline rebuild can tell
    judge spend from target spend -- and so a run's cost is not understated by
    however many ambiguities it happened to have."""

    @property
    def resolved(self) -> int:
        return len(self.observations)


def _body(judge: JudgeConfig, prompt: str) -> dict[str, Any]:
    """An OpenAI-shaped chat request at temperature 0.

    Pinned model, zero temperature, fixed prompt: §11.9's three determinism
    requirements, all of which reach the manifest via
    ``judge{model, prompt_version}``.
    """
    return {
        "model": judge.model,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }


async def aresolve(
    escalations: list[Escalation],
    *,
    judge: JudgeConfig,
    client: TransportClient,
    texts: dict[tuple[str, int], str],
    config_id: str,
) -> JudgeOutcome:
    """Ask the judge about each ambiguity. One call per escalation.

    ``texts`` supplies the response being judged, keyed by
    ``(unit_id, run_idx)`` -- read from the blob store rather than held in
    memory, so a resumed run judges the same bytes the original scored.
    """
    outcome = JudgeOutcome()
    headers = {"Authorization": f"Bearer {judge.key}"} if judge.key else {}

    for escalation in escalations:
        rubric = rubric_for(escalation.contract_kind)
        response_text = texts.get(
            (escalation.observation.unit_id, escalation.observation.run_idx), ""
        )
        if rubric is None or not response_text:
            # `plan_escalations` filters both, so reaching here means the plan
            # and the texts disagree. Recorded rather than assumed away.
            outcome.failures.append(
                (escalation.observation.unit_id, "no rubric or no response text")
            )
            continue

        prompt = build_prompt(
            rubric,
            probe=escalation.probe_text(),
            response=response_text,
            expectation=escalation.expectation,
        )

        try:
            results = await client.call(
                judge.url,
                _body(judge, prompt),
                config_id=config_id,
                unit_id=escalation.observation.unit_id,
                run_idx=escalation.observation.run_idx,
                role="judge",
                headers=headers,
            )
        except Exception as error:
            outcome.failures.append(
                (escalation.observation.unit_id, f"{type(error).__name__}: {error}")
            )
            continue

        outcome.requests += len(results)
        outcome.tokens_out += sum(r.call.tokens.out or 0 for r in results)
        outcome.calls.extend(results)
        final = results[-1] if results else None
        if final is None:
            outcome.failures.append((escalation.observation.unit_id, "no response"))
            continue

        try:
            verdict = parse_verdict(_reply_text(final))
        except JudgeError as error:
            # The deterministic UNSCORABLE stands. A judge that cannot answer
            # must not be rounded into one that did.
            outcome.failures.append((escalation.observation.unit_id, str(error)))
            continue

        outcome.observations.append(
            resolved_observation(escalation, verdict, judge)
        )

    return outcome


def _reply_text(result: CallResult) -> str:
    """The judge's message content, out of the raw body.

    `TransportClient` populates `text` only for streamed responses; for
    everything else extraction belongs to the caller, which is why the runner
    does the same thing for target calls.
    """
    import json as _json

    if result.text:
        return result.text
    try:
        payload = _json.loads(result.raw_body or b"{}")
    except _json.JSONDecodeError:
        return ""
    return extract_at(payload, JUDGE_TEXT_PATH) or ""
