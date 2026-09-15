"""Baseline and gate, end to end (spec §16)."""

from __future__ import annotations

from pathlib import Path

from tests.conftest import make_app, make_client

from sweepeval.execute.evaluate import aevaluate_target
from sweepeval.execute.gate import (
    DEFAULT_GATE_ON,
    gate,
    load_baseline,
    save_baseline,
    snapshot,
)
from sweepeval.stats.diff import ExitCode


async def _evaluate(scenario: str, tmp_path: Path, profile: str = "standard", **kw):
    app = make_app(scenario)
    client = make_client(app)
    try:
        return await aevaluate_target(
            "https://mock.test" + app.scenario.paths[0],
            client=client, root=str(tmp_path), runs=2, profile=profile,  # type: ignore[arg-type]
            authorized=True, authorization_prompt=False, seed=7, **kw,
        )
    finally:
        await client.aclose()


# --- snapshot and round trip ----------------------------------------------


async def test_a_baseline_round_trips_through_disk(tmp_path: Path) -> None:
    result = await _evaluate("openai_clean", tmp_path, key="test-key-abcdefgh")
    path = save_baseline(snapshot(result), tmp_path / "baseline.json")
    restored = load_baseline(path)

    assert restored.run_id == result.run_id
    assert restored.comparability.hard.corpus_hash == result.comparability.hard.corpus_hash


async def test_the_baseline_carries_per_cluster_values(tmp_path: Path) -> None:
    """§16: the paired test needs them; a point and an interval throw away the
    pairing that stops the gate flapping."""
    result = await _evaluate("openai_clean", tmp_path, key="test-key-abcdefgh")
    baseline = snapshot(result)
    assert baseline.clusters["security_pass_rate"]
    assert len(baseline.clusters["security_pass_rate"]) > 1


async def test_the_baseline_contains_no_credential(tmp_path: Path) -> None:
    """It is designed to be committed (§6.6)."""
    key = "sk-supersecretkeyvalue"
    app = make_app("query_param_auth")
    client = make_client(app)
    try:
        result = await aevaluate_target(
            "https://mock.test/v1/chat/completions", key=key, client=client,
            root=str(tmp_path), runs=1, profile="standard",
            authorized=True, authorization_prompt=False, seed=7,
        )
    finally:
        await client.aclose()

    path = save_baseline(snapshot(result), tmp_path / "baseline.json")
    assert key not in path.read_text(encoding="utf-8")


async def test_a_committed_baseline_is_sorted_for_clean_diffs(tmp_path: Path) -> None:
    result = await _evaluate("openai_clean", tmp_path, key="test-key-abcdefgh")
    text = save_baseline(snapshot(result), tmp_path / "b.json").read_text(encoding="utf-8")
    import json

    payload = json.loads(text)
    for values in payload["clusters"].values():
        assert list(values) == sorted(values)


# --- the gate --------------------------------------------------------------


async def test_an_unchanged_target_passes_the_gate(tmp_path: Path) -> None:
    """The case that decides whether anyone leaves the gate switched on."""
    first = await _evaluate("openai_clean", tmp_path / "a", key="test-key-abcdefgh")
    baseline = snapshot(first)
    second = await _evaluate("openai_clean", tmp_path / "b", key="test-key-abcdefgh")

    verdict = gate(second, baseline, seed=1)
    assert verdict.ok, verdict.explain()
    assert verdict.exit_code is ExitCode.PASS


async def test_a_real_regression_fires(tmp_path: Path) -> None:
    """A target that starts leaking must fail the build."""
    clean = await _evaluate("openai_clean", tmp_path / "a", key="test-key-abcdefgh")
    baseline = snapshot(clean)
    leaky = await _evaluate("leaky_guardrails", tmp_path / "b")

    verdict = gate(leaky, baseline, seed=1, gate_on=("security_pass_rate",))
    # Comparability refuses first if the extraction path or target type moved,
    # which is itself correct behaviour — assert we get a non-pass either way.
    assert not verdict.ok
    assert verdict.exit_code in (ExitCode.REGRESSION, ExitCode.COMPARABILITY_REFUSED)


async def test_latency_is_excluded_from_the_default_gate(tmp_path: Path) -> None:
    """§16: between-session variance is far above what we can attribute to the
    target, so gating latency flaps for reasons unrelated to the code."""
    assert "latency_p95_ms" not in DEFAULT_GATE_ON
    assert "security_pass_rate" in DEFAULT_GATE_ON


# --- comparability refuses before any statistics (I6) ---------------------


async def test_a_corpus_change_refuses_rather_than_reporting_a_regression(
    tmp_path: Path,
) -> None:
    """"These measured different things" is not "this got worse", and
    reporting the first as the second teaches people to ignore the gate."""
    result = await _evaluate("openai_clean", tmp_path, key="test-key-abcdefgh")
    baseline = snapshot(result)

    tampered = baseline.model_copy(
        update={
            "comparability": baseline.comparability.model_copy(
                update={
                    "hard": baseline.comparability.hard.model_copy(
                        update={"corpus_hash": "f" * 64}
                    )
                }
            )
        }
    )

    verdict = gate(result, tampered, seed=1)
    assert not verdict.ok
    assert verdict.exit_code is ExitCode.COMPARABILITY_REFUSED
    assert any("corpus hash differs" in r for r in verdict.refusals)
    assert any("not a regression" in n for n in verdict.notes)


async def test_a_scorer_version_bump_refuses(tmp_path: Path) -> None:
    """A patched scorer changes what its rate counts (§6.5)."""
    result = await _evaluate("openai_clean", tmp_path, key="test-key-abcdefgh")
    baseline = snapshot(result)
    bumped = dict(baseline.comparability.hard.scorer_versions)
    bumped["security"] = bumped["security"] + 1

    tampered = baseline.model_copy(
        update={
            "comparability": baseline.comparability.model_copy(
                update={
                    "hard": baseline.comparability.hard.model_copy(
                        update={"scorer_versions": bumped}
                    )
                }
            )
        }
    )
    assert gate(result, tampered, seed=1).exit_code is ExitCode.COMPARABILITY_REFUSED


async def test_the_quick_profile_is_refused_for_gating(tmp_path: Path) -> None:
    """§16: quick's intervals are wide by design. Passing everything and
    calling it green would be worse than refusing."""
    first = await _evaluate(
        "openai_clean", tmp_path / "a", profile="quick", key="test-key-abcdefgh"
    )
    second = await _evaluate(
        "openai_clean", tmp_path / "b", profile="quick", key="test-key-abcdefgh"
    )

    # Both quick, so comparability passes and the eligibility check is what
    # fires. A quick-vs-standard pair would be refused by I6 first, which is
    # also correct but tests a different rule.
    verdict = gate(second, snapshot(first), seed=1)
    assert not verdict.ok
    assert verdict.exit_code is ExitCode.USAGE_ERROR
    assert any("not gate-eligible" in r for r in verdict.refusals)


# --- the payload the Action consumes --------------------------------------


async def test_the_gate_payload_is_machine_readable(tmp_path: Path) -> None:
    from sweepeval.execute.gate import gate_payload

    first = await _evaluate("openai_clean", tmp_path / "a", key="test-key-abcdefgh")
    second = await _evaluate("openai_clean", tmp_path / "b", key="test-key-abcdefgh")
    payload = gate_payload(gate(second, snapshot(first), seed=1))

    assert set(payload) == {
        "ok", "exit_code", "regressions", "metrics", "hard_fails", "refusals",
        # What the gate could NOT do. A consumer that cannot see this cannot
        # tell a run that tested five metrics from one that tested one.
        "not_gated", "degraded", "incomplete",
        # Which models the verdict is about. A gate log read three weeks later
        # could not say, and the gate will refuse on a mismatch.
        "model",
    }
    assert isinstance(payload["exit_code"], int)
    assert set(payload["model"]) == {"baseline", "current"}
