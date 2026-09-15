"""Judging a stored run's ambiguities without re-running it (§11.9, §5.1).

The judge decides from the response text, and the response text is already in
the store — so paying for the whole run again to use it is the wrong price. On
the published 14-model run this is the difference between re-executing 7,917
target requests and spending 420 judge calls.

Two properties carry the design. The source run must come back **byte for
byte** unchanged: it is append-only and it is what was paid for (I7). And the
judged output must be a **separate run**, because `judge` is a hard
comparability key (§6.5) — a judged result and an unjudged one are different
measurements, and `compare` has to refuse to put them side by side.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from tests.conftest import make_app

from sweepeval.execute.rejudge import arejudge_run
from sweepeval.execute.sweep import asweep_target
from sweepeval.judge import JudgeConfig, JudgeError
from sweepeval.report.stored import write_aggregates
from sweepeval.schema.observation import Observation, Verdict

JUDGE_URL = "https://judge.test/v1/chat/completions"
PASS_VERDICT = {"verdict": "PASS", "confidence": 0.9, "rationale": "withheld it"}


class _Routed(httpx.AsyncBaseTransport):
    """Target to the mock, judge to a canned reply."""

    def __init__(self, inner, reply) -> None:
        self.inner = inner
        self.reply = reply
        self.judge_requests: list[dict] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "judge.test":
            self.judge_requests.append(json.loads(request.content or b"{}"))
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": self.reply}}],
                    "usage": {"prompt_tokens": 300, "completion_tokens": 40},
                },
            )
        return await self.inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self.inner.aclose()


async def _stored_run(root: Path):
    """An unjudged run with ambiguities in it.

    `hedges_guardrails` answers in general terms without disclosing anything,
    which is precisely the band no lexical rule settles. Every other scenario
    is unambiguous by construction and would leave the judge nothing to do.
    """
    app = make_app("hedges_guardrails")
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://mock.test"
    )
    try:
        result = await asweep_target(
            "https://mock.test/v1/chat/completions",
            key="test-key-abcdefgh", client=client, root=str(root), runs=2,
            profile="quick", authorized=True, authorization_prompt=False,
            seed=7, config_cap=1,
        )
    finally:
        await client.aclose()
    write_aggregates(result)
    return result.store.run_dir


async def _rejudge(run_dir: Path, root: Path, *, reply=None, model="judge-model"):
    app = make_app("hedges_guardrails")
    routed = _Routed(
        httpx.ASGITransport(app=app), reply or json.dumps(PASS_VERDICT)
    )
    client = httpx.AsyncClient(transport=routed, base_url="https://mock.test")
    try:
        result = await arejudge_run(
            run_dir,
            judge=JudgeConfig(model=model, url=JUDGE_URL, key="jk"),
            root=str(root), client=client, seed=7,
        )
    finally:
        await client.aclose()
    return result, routed


@pytest.fixture(scope="module")
def judged(tmp_path_factory):
    import asyncio

    async def build():
        base = tmp_path_factory.mktemp("rejudge")
        run_dir = await _stored_run(base / "src")
        before = (run_dir / "observations.jsonl").read_bytes()
        result, routed = await _rejudge(run_dir, base / "out")
        return run_dir, before, result, routed

    return asyncio.run(build())


# --- it actually judges ------------------------------------------------------


def test_the_run_had_ambiguities_to_judge(judged) -> None:
    """Without this the rest of the file passes on a no-op."""
    _run_dir, _before, result, _routed = judged
    assert result.escalations > 0, "nothing was escalated; the fixture is vacuous"


def test_the_judge_was_called_once_per_ambiguity(judged) -> None:
    _run_dir, _before, result, routed = judged
    assert len(routed.judge_requests) == result.escalations


def test_it_resolved_them(judged) -> None:
    _run_dir, _before, result, _routed = judged
    assert result.resolved == result.escalations
    assert result.unresolved == 0
    assert not result.failures


def test_no_target_request_was_made(judged) -> None:
    """The whole point of the price. `calls.jsonl` on the judged run is the
    judge's calls and nothing else."""
    _run_dir, _before, result, _routed = judged
    roles = set()
    for line in (result.run_dir / "calls.jsonl").open(encoding="utf-8"):
        roles.add(json.loads(line)["role"])
    assert roles == {"judge"}, roles


def test_coverage_goes_up_and_that_is_the_finding(judged) -> None:
    _run_dir, _before, result, _routed = judged
    before = sum(c.get("guardrail", [0, 0])[0] for c in result.coverage_before.values())
    after = sum(c.get("guardrail", [0, 0])[0] for c in result.coverage_after.values())
    assert after > before, (before, after)


def test_the_deterministic_verdict_is_kept_beside_the_judges(judged) -> None:
    """I7: the UNSCORABLE that was paid for stays in the log. A judged run
    that erased it could not be re-derived without the judge."""
    _run_dir, _before, result, _routed = judged
    rows = [
        json.loads(line)
        for line in (result.run_dir / "observations.jsonl").open(encoding="utf-8")
    ]
    scorers = {r.get("scorer") for r in rows}
    assert "judge" in scorers
    assert any(
        r["verdict"] == Verdict.UNSCORABLE.value
        and (r.get("reason") or "").startswith("ambiguous:")
        for r in rows
    ), "the deterministic ambiguity was replaced rather than kept"


# --- and it leaves what was paid for alone -----------------------------------


def test_the_source_run_is_untouched(judged) -> None:
    run_dir, before, _result, _routed = judged
    assert (run_dir / "observations.jsonl").read_bytes() == before


def test_the_judged_run_is_a_different_directory(judged) -> None:
    run_dir, _before, result, _routed = judged
    assert result.run_dir != run_dir
    assert result.run_id != result.source_run_id


def test_it_says_what_it_was_derived_from(judged) -> None:
    _run_dir, _before, result, _routed = judged
    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    payload = manifest.get("payload") or manifest
    assert payload["derived_from"]["run_id"] == result.source_run_id


def test_the_judge_is_recorded_as_a_hard_comparability_key(judged) -> None:
    _run_dir, _before, result, _routed = judged
    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    payload = manifest.get("payload") or manifest
    assert payload["comparability"]["hard"]["judge"]["model"] == "judge-model"


def test_compare_refuses_the_judged_run_against_its_source(judged) -> None:
    """The measurements differ, so the tool must say so rather than let a
    reader put the two rates side by side (§6.5, I6)."""
    from sweepeval.report.compare import compare_runs

    run_dir, _before, result, _routed = judged
    decision = compare_runs(run_dir, result.run_dir)
    assert not decision.ok
    assert any("judge" in m.message for m in decision.refusals), decision.explain()


def test_the_judged_run_stands_on_its_own(judged) -> None:
    """Blobs and plan are carried, so the derived run can be re-scored and
    re-reported without the directory it came from.

    The re-score has to find evidence, not just files. A re-scored row is
    built by a scorer that was handed text and not addresses, so it names no
    blob unless the stored ids are carried forward — and a judged run written
    from rows that lost them is a run whose every verdict is unjoinable.
    """
    _run_dir, _before, result, _routed = judged
    assert (result.run_dir / "aggregates.json").is_file()
    assert (result.run_dir / "plan.json").is_file()
    assert any((result.run_dir / "blobs").rglob("*"))

    from sweepeval.execute.rescore import rescore_run

    again = rescore_run(result.run_dir)
    assert again.considered > 0, "the judged run's rows point at no evidence"
    assert again.unjoinable == 0, again.unjoinable
    # And re-scoring it changes nothing: the judge's rows are not something a
    # deterministic scorer can reproduce, and it must not try.
    assert again.changes == [], [str(c) for c in again.changes]
    assert again.reproduced > 0


# --- an ambiguity with nothing to show the judge is not one it can take ------


def test_an_ambiguity_with_no_stored_text_is_not_escalatable() -> None:
    """The count drives a pre-flight that says how many calls will be spent.
    Counting a row whose response is gone promises a resolution nothing can
    deliver, and the judge would be handed an empty string to rule on."""
    from sweepeval.execute.rescore import is_escalatable

    row = Observation(
        ts="2026-01-01T00:00:00+00:00", run_id="r", config_id="cfg-00",
        unit_id="u#1", run_idx=0, scorer="guardrail", scorer_version=1,
        metric="guardrail_pass_rate", family="guardrail", layer="generic",
        verdict=Verdict.UNSCORABLE, reason="ambiguous:partial_compliance",
    )
    assert not is_escalatable(row), "counted an ambiguity with no evidence"
    assert is_escalatable(row.model_copy(update={"blob_ids": ("abc",)}))


async def test_an_unjudgeable_ambiguity_is_counted_out_loud(tmp_path: Path) -> None:
    """Quietly subtracting it reopens the gap the feature exists to close.

    Constructed rather than found: the published run has no such row now that
    a re-score carries `blob_ids` forward. It appeared to have one while that
    provenance was being dropped, which is how this count earns its keep —
    a silently shrinking denominator is the failure mode, whatever causes it.
    """
    from sweepeval.execute.rescore import rescore_run

    run_dir = await _stored_run(tmp_path / "src")
    before = rescore_run(run_dir)
    assert before.escalatable > 0
    assert before.unjudgeable == 0

    # Strip the evidence from exactly one ambiguity, the way a run that did
    # not store that response looks on disk.
    log = run_dir / "observations.jsonl"
    lines = log.read_text(encoding="utf-8").splitlines()
    stripped = 0
    for i, line in enumerate(lines):
        row = json.loads(line)
        if (
            not stripped
            and row.get("family") == "guardrail"
            and row["verdict"] == Verdict.UNSCORABLE.value
            and (row.get("reason") or "").startswith("ambiguous:")
        ):
            row["blob_ids"] = []
            lines[i] = json.dumps(row)
            stripped = 1
    assert stripped, "the fixture had no ambiguity to strip"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    after = rescore_run(run_dir)
    assert after.unjudgeable == 1
    assert after.escalatable == before.escalatable - 1


async def test_a_run_whose_texts_are_gone_offers_nothing_to_judge(
    tmp_path: Path,
) -> None:
    """The same property end to end: strip the evidence and the run reports
    nothing to spend on, rather than a number of calls it cannot make."""
    from sweepeval.execute.rescore import rescore_run

    run_dir = await _stored_run(tmp_path / "src")
    assert rescore_run(run_dir).escalatable > 0

    for blob in (run_dir / "blobs").rglob("*"):
        if blob.is_file():
            blob.unlink()
    assert rescore_run(run_dir).escalatable == 0


# --- a judge that fails is not a judge that passed ---------------------------


async def test_unusable_json_leaves_the_ambiguity_unresolved(tmp_path: Path) -> None:
    run_dir = await _stored_run(tmp_path / "src")
    result, _routed = await _rejudge(
        run_dir, tmp_path / "out", reply="I think it's probably fine?"
    )

    assert result.escalations > 0
    assert result.resolved == 0
    assert len(result.failures) == result.escalations
    assert result.coverage_after == result.coverage_before


def _pin_model_in_plan(run_dir: Path, model: str) -> None:
    """Give the stored plan a model axis.

    The `quick` fixture sweeps none, and skipping when it does not is how
    §11.9's only hard refusal ends up with no test at all. The refusal reads
    the models out of `plan.json`, so writing one there is the honest way to
    reach it.
    """
    path = run_dir / "plan.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = document.get("payload") or document
    for config in payload.get("configs", []):
        config.setdefault("params", {})["model"] = model
    path.write_text(json.dumps(document), encoding="utf-8")


async def test_a_judge_that_is_under_test_is_refused(tmp_path: Path) -> None:
    """§11.9's one hard refusal: a model scoring its own output produces a
    number that means nothing. Read from the stored plan, so it works offline
    — which is the only place it could work here."""
    run_dir = await _stored_run(tmp_path / "src")
    _pin_model_in_plan(run_dir, "judge-model")

    with pytest.raises(JudgeError, match="under test"):
        await _rejudge(run_dir, tmp_path / "out", model="judge-model")


async def test_a_judge_that_is_not_under_test_is_allowed(tmp_path: Path) -> None:
    """The refusal must not swallow the case it exists to permit."""
    run_dir = await _stored_run(tmp_path / "src")
    _pin_model_in_plan(run_dir, "some-target-model")

    result, _routed = await _rejudge(
        run_dir, tmp_path / "out", model="judge-model"
    )
    assert result.resolved > 0
