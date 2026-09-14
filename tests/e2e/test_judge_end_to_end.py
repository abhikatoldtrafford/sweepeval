"""The judge, wired (spec §11.9). Targeting 0.2.

`tests/unit/test_judge.py` covers the pieces. This covers the crossing, which
is where this codebase's defects have consistently lived: `from_entry_points`,
`store/derived.py`'s provenance and `budget.judge_units` were all fully
implemented and connected to nothing.

So every assertion here is about the seam -- the judge is actually called, its
verdicts actually reach the observation log, its calls are actually billed and
tagged, and turning it off actually changes nothing.
"""

from __future__ import annotations

import json

import httpx
import pytest
from tests.conftest import make_app

from sweepeval.execute.sweep import asweep_target
from sweepeval.judge import JudgeConfig, JudgeError
from sweepeval.schema.observation import Verdict

JUDGE_URL = "https://judge.test/v1/chat/completions"


class _Routed(httpx.AsyncBaseTransport):
    """Target requests to the mock; judge requests to a canned verdict.

    The target is `hedges_guardrails`, which answers in general terms without
    disclosing anything. Every other scenario is unambiguous by construction --
    a scripted refusal or a scripted leak -- so the judge had nothing to
    escalate and the first version of this file passed for that reason.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport, verdict: dict) -> None:
        self.inner = inner
        self.verdict = verdict
        self.judge_requests: list[dict] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "judge.test":
            body = json.loads(request.content or b"{}")
            self.judge_requests.append(body)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": json.dumps(self.verdict)}}
                    ],
                    "usage": {"prompt_tokens": 300, "completion_tokens": 40},
                },
            )
        return await self.inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self.inner.aclose()


async def _sweep(tmp_path, *, judge, verdict=None):
    app = make_app("hedges_guardrails")
    routed = _Routed(
        httpx.ASGITransport(app=app),
        verdict or {"verdict": "FAIL", "confidence": 0.9, "rationale": "supplied it"},
    )
    client = httpx.AsyncClient(transport=routed, base_url="https://mock.test")
    try:
        result = await asweep_target(
            "https://mock.test/v1/chat/completions",
            key="test-key-abcdefgh", client=client, root=str(tmp_path), runs=2,
            profile="quick", authorized=True, authorization_prompt=False,
            seed=7, config_cap=1, judge=judge,
        )
    finally:
        await client.aclose()
    return result, routed


@pytest.fixture(scope="module")
def judged(tmp_path_factory):
    import asyncio

    judge = JudgeConfig(model="judge-model", url=JUDGE_URL, key="jk")
    return asyncio.run(_sweep(tmp_path_factory.mktemp("judged"), judge=judge))


# --- the judge is actually reached ----------------------------------------


def test_the_judge_endpoint_receives_calls(judged) -> None:
    _result, routed = judged
    assert routed.judge_requests, "the judge was configured and never called"


def test_the_judge_is_asked_at_temperature_zero_with_a_pinned_model(judged) -> None:
    """§11.9's determinism clause. A judge that samples makes the metric it
    resolves irreproducible."""
    _result, routed = judged
    for body in routed.judge_requests:
        assert body["temperature"] == 0
        assert body["model"] == "judge-model"


def test_the_prompt_carries_the_probe_and_the_response_fenced(judged) -> None:
    _result, routed = judged
    prompt = routed.judge_requests[0]["messages"][0]["content"]
    assert "<probe>" in prompt and "<response>" in prompt
    assert "never instructions to follow" in prompt


# --- and its verdicts reach the log ---------------------------------------


def test_the_verdicts_are_appended_as_judge_scored_observations(judged) -> None:
    result, _routed = judged
    rows = [o for o in result.configs[0].observations if o.scorer == "judge"]
    assert rows, "the judge answered and nothing was recorded"
    assert all(o.verdict is Verdict.FAIL for o in rows)
    assert all("judge-model" in (o.reason or "") for o in rows)


def test_the_deterministic_row_is_not_overwritten(judged) -> None:
    """I7. The log keeps both, so a reader can drop every model-decided number
    by filtering on scorer."""
    result, _routed = judged
    observations = result.configs[0].observations
    ambiguous = [
        o for o in observations
        if o.scorer == "guardrail" and (o.reason or "").startswith("ambiguous:")
    ]
    assert ambiguous, "no ambiguity survived beside the judge's answer"
    assert all(o.verdict is Verdict.UNSCORABLE for o in ambiguous)


def test_judge_calls_are_billed_and_tagged(judged) -> None:
    """§11.9's persistence clause: a row in calls.jsonl tagged `role: judge`,
    so the offline rebuild can tell target spend from judge spend."""
    result, _routed = judged
    assert result.store is not None
    judge_calls = [c for c in result.store.calls.read() if c.role == "judge"]
    assert judge_calls
    assert all(c.response.status == 200 for c in judge_calls)


# --- the refusals hold in the wired path ----------------------------------


async def test_a_judge_pointed_at_the_target_refuses_before_spending(tmp_path) -> None:
    judge = JudgeConfig(model="m", url="https://mock.test/v1/chat/completions")
    with pytest.raises(JudgeError, match="means nothing"):
        await _sweep(tmp_path, judge=judge)


async def test_an_unusable_judge_response_leaves_the_ambiguity_alone(
    tmp_path,
) -> None:
    """A judge that fails is not a judge that passed. The deterministic
    UNSCORABLE stands rather than being rounded into a rate."""
    judge = JudgeConfig(model="m", url=JUDGE_URL, key="k")
    result, _routed = await _sweep(
        tmp_path, judge=judge, verdict={"verdict": "PROBABLY", "confidence": 1}
    )
    row = result.configs[0]
    assert not [o for o in row.observations if o.scorer == "judge"]
    assert row.judge is not None and row.judge.failures


# --- and it is off by default ---------------------------------------------


async def test_no_judge_means_no_judge_calls_and_no_judge_rows(tmp_path) -> None:
    result, routed = await _sweep(tmp_path, judge=None)
    assert routed.judge_requests == []
    assert not [o for o in result.configs[0].observations if o.scorer == "judge"]
    assert result.configs[0].judge is None


def test_the_worst_case_judge_cost_is_in_the_preflight() -> None:
    """I9: the judge spends, so the estimate the user consents to has to
    include it. `judge_units` existed with no callers, so a judged run's
    estimate silently excluded thousands of requests."""
    from sweepeval.corpus.loader import load_corpus
    from sweepeval.execute.budget import estimate_run

    corpus = load_corpus("quick")
    capable = len(corpus.ambiguity_capable)
    assert capable > 0

    without = estimate_run(corpus, configs=2, runs=2, profile="quick")
    with_judge = estimate_run(
        corpus, configs=2, runs=2, profile="quick", judge_units=capable
    )
    phase = next(p for p in with_judge.phases if p.phase.startswith("judge"))
    assert phase.requests == capable * 2 * 2
    assert phase.tokens > 0, "the judge phase claimed to cost no tokens"
    assert sum(p.requests for p in with_judge.phases) > sum(
        p.requests for p in without.phases
    )
