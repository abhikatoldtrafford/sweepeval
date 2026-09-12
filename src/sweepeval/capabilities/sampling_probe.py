"""Sending the sampling-effect test (spec §9.1).

:mod:`sweepeval.capabilities.sampling` owns the decision table and the
statistics. This module owns the requests: it sends each instrument prompt at
a low and a high setting, hands the outputs to the decision functions, and
charges every call to the capability budget.

The separation matters because the decision table is where the interesting
mistakes live, and a pure function is testable without a server. What is left
here is plumbing, and it is written to fail loudly rather than guess:

* A parameter the budget could not afford is **not tested and not swept**, and
  the planner prints why. Reporting it as INCONCLUSIVE would sweep an axis on
  no evidence and triple the run's cost on a coin flip.
* A parameter the target rejects outright (a 400 on every probe) is likewise
  not an axis. A sweep over a parameter the endpoint refuses is a sweep over
  identical failures.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from sweepeval.capabilities.sampling import (
    SamplingVerdict,
    Verdict,
    decide_tier1,
    decide_tier2,
)
from sweepeval.discovery.budget import Attempt, DiscoveryBudget
from sweepeval.discovery.extract import extract_at
from sweepeval.discovery.ladder import LadderResult, body_for_turns

__all__ = [
    "RUNS_PER_SETTING",
    "SETTINGS",
    "SamplingReport",
    "probe_sampling",
]

SETTINGS: dict[str, tuple[float, float]] = {
    "temperature": (0.0, 1.0),
    "top_p": (0.1, 1.0),
}
"""Low and high ends per parameter. Fixed by the spec, not exposed as a flag:
a user-chosen pair would make INERT mean something different in every run."""

RUNS_PER_SETTING = 3
"""§9.1's decision table is defined on distinct-output counts out of three.
Changing this changes what every cell of that table means."""


@dataclass
class SamplingReport:
    """Verdicts for the parameters that were tested, and why the rest were not."""

    verdicts: dict[str, SamplingVerdict] = field(default_factory=dict)
    not_tested: dict[str, str] = field(default_factory=dict)
    requests: int = 0

    def swept(self) -> tuple[str, ...]:
        return tuple(sorted(p for p, v in self.verdicts.items() if v.swept))


async def probe_sampling(
    client: httpx.AsyncClient,
    ladder: LadderResult,
    key: str | None,
    text_path: str | None,
    prompts: Sequence[str],
    *,
    budget: DiscoveryBudget,
    parameters: Sequence[str] = ("temperature", "top_p"),
    seed: int = 0,
) -> SamplingReport:
    """Test each parameter with the corpus's open-ended instrument prompts."""
    report = SamplingReport()

    if not prompts:
        for parameter in parameters:
            report.not_tested[parameter] = (
                "the corpus supplied no open-ended instrument prompt, and a short "
                "factual prompt answers identically at any setting"
            )
        return report

    for parameter in parameters:
        if parameter not in SETTINGS:
            report.not_tested[parameter] = f"no low/high pair is defined for {parameter}"
            continue

        needed = 2 * len(prompts) * RUNS_PER_SETTING
        if budget.remaining_posts() < needed:
            report.not_tested[parameter] = (
                f"capability budget has {budget.remaining_posts()} request(s) left "
                f"and the test needs {needed}; the axis is not swept, because "
                f"sweeping it on no evidence would triple the run's cost on a "
                f"coin flip"
            )
            continue

        low_value, high_value = SETTINGS[parameter]
        low_runs: list[list[str]] = []
        high_runs: list[list[str]] = []
        rejected = False

        for prompt in prompts:
            low, low_ok = await _repeat(
                client, ladder, key, budget, text_path, prompt, parameter, low_value
            )
            high, high_ok = await _repeat(
                client, ladder, key, budget, text_path, prompt, parameter, high_value
            )
            if not (low_ok and high_ok):
                rejected = True
                break
            low_runs.append(low)
            high_runs.append(high)

        if rejected or not low_runs:
            report.not_tested[parameter] = (
                f"the target did not answer the {parameter} probes (rejected the "
                f"parameter, or the budget ran out mid-test)"
            )
            continue

        tier1 = decide_tier1(low_runs, high_runs)
        if tier1.verdict is Verdict.EFFECTIVE:
            report.verdicts[parameter] = SamplingVerdict(
                parameter=parameter,
                verdict=Verdict.EFFECTIVE,
                tier=1,
                evidence=tier1.evidence,
                reason=tier1.reason,
            )
            continue

        # Tier 1 could not separate, so pool every run of every prompt and let
        # the dispersion test decide. Pooling is what gives tier 2 enough
        # sample to say INERT rather than shrug (§9.1).
        report.verdicts[parameter] = decide_tier2(
            parameter,
            [text for runs in low_runs for text in runs],
            [text for runs in high_runs for text in runs],
            seed=seed,
        )

    report.requests = budget.posts
    return report


async def _repeat(
    client: httpx.AsyncClient,
    ladder: LadderResult,
    key: str | None,
    budget: DiscoveryBudget,
    text_path: str | None,
    prompt: str,
    parameter: str,
    value: float,
) -> tuple[list[str], bool]:
    """Send one prompt ``RUNS_PER_SETTING`` times at one setting."""
    from sweepeval.discovery.auth import apply_auth

    headers, params = apply_auth(ladder.auth, key)
    # Through body_for_turns: a mutated shape needs whatever the mutation
    # added. Built from the shape alone, every sampling probe against OpenAI
    # 400'd and both temperature and top_p were reported as "the target did
    # not answer", which reads as a property of the target rather than a
    # malformed request.
    body = body_for_turns(ladder, [("user", prompt)], **{parameter: value})

    texts: list[str] = []
    for _ in range(RUNS_PER_SETTING):
        if budget.exhausted():
            return texts, False
        status, payload = await _post(client, ladder.path, body, headers, params)
        budget.record(
            Attempt(
                stage="capability",
                shape=f"sampling:{parameter}={value}",
                path=httpx.URL(ladder.path).path,
                mutation=None,
                status=status,
                content_type="application/json",
                error_excerpt="",
            ),
            tokens=300,
        )
        if status >= 400 or payload is None:
            return texts, False
        texts.append(_text_of(payload, text_path))

    # An empty answer is not a measurement of dispersion; three of them would
    # read as perfectly deterministic and mark a live axis INERT.
    return texts, all(t.strip() for t in texts)


async def _post(
    client: httpx.AsyncClient,
    path: str,
    body: dict[str, Any],
    headers: dict[str, str],
    params: dict[str, Any],
) -> tuple[int, Any]:
    try:
        response = await client.post(path, json=body, headers=headers, params=params)
    except httpx.HTTPError:
        return 0, None
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, None


def _text_of(payload: Any, path: str | None) -> str:
    if payload is None or path is None:
        return ""
    return extract_at(payload, path) or ""
