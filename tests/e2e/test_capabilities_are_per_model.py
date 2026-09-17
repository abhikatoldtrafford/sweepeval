"""Capabilities belong to the model, not to the run (spec §9, §12).

Detection ran once per sweep and the answer was applied to every config. A
sweep's whole purpose is varying the model, and capabilities vary with it, so
every capability-gated family was gated on a verdict measured against whichever
model discovery happened to pick.

Measured, on a live four-model run: `gpt-5-search-api` was the only model in it
with retrieval, and it was told it had none -- the family was skipped for all
four configs and the run produced zero retrieval rows, which was the entire
reason that model was in it. Symmetrically it was told it had tool calling,
which it rejects, so 36 tool probes went out and came back 404 and its tool
rows read UNSCORABLE rather than honestly skipped.
"""

from __future__ import annotations

import tempfile

from tests.conftest import make_app, make_client

from sweepeval.capabilities.detect import Capability, Support
from sweepeval.execute.declared import DeclaredConfig
from sweepeval.execute.sweep import asweep_target


async def _sweep(
    scenario: str, cap: int = 2, profile: str = "quick",
    models: list[str] | None = None,
):
    """A sweep whose model axis the planner builds from the mock's own list.

    `demands_a_model` exposes two model ids, so the planner produces a model
    axis without anything being declared -- which is the situation the defect
    lived in.
    """
    app = make_app(scenario)
    client = make_client(app)
    try:
        return await asweep_target(
            "https://mock.test" + app.scenario.paths[0],
            key="test-key-abcdefgh", client=client, root=tempfile.mkdtemp(),
            runs=2, profile=profile, authorized=True,
            authorization_prompt=False, seed=7, config_cap=cap,
            declared=DeclaredConfig(axes={"model": models}) if models else None,
        )
    finally:
        await client.aclose()


async def test_each_config_carries_its_own_capability_report() -> None:
    """The fix, in the artifact. Before it, `ConfigResult.capabilities` was a
    default-constructed empty report on every row: the only capability verdict
    in the run lived at run level, where it could not vary by model."""
    result = await _sweep(
        "demands_a_model", cap=2,
        models=["gpt-mock-large", "gpt-mock-small"],
    )
    models = {r.config.params.get("model") for r in result.configs}
    assert len(models) >= 2, models
    for row in result.configs:
        assert row.capabilities.results, (
            f"{row.config.config_id} carries no capability report of its own"
        )
        assert Capability.TOOL_CALLING in row.capabilities.results


async def test_a_config_that_pins_no_model_reuses_the_run_level_report() -> None:
    """Re-probing would spend for an answer already known: nothing about the
    target changed."""
    result = await _sweep("openai_clean", cap=1)
    assert len(result.configs) >= 1
    row = result.configs[0]
    assert not row.config.params.get("model")
    assert row.capabilities is result.capabilities


async def test_the_estimate_prices_one_capability_phase_per_model() -> None:
    """I9: detecting per model costs more requests, and the user consents to
    the estimate before anything is sent."""
    from sweepeval.corpus.loader import load_corpus
    from sweepeval.execute.budget import estimate_run

    corpus = load_corpus("standard")
    one = estimate_run(corpus, configs=4, runs=3, profile="standard",
                       capability_phases=1)
    four = estimate_run(corpus, configs=4, runs=3, profile="standard",
                        capability_phases=4)
    assert four.total_requests > one.total_requests

    extra = next(
        p for p in four.phases if p.phase == "capabilities (per extra model)"
    )
    assert "3 further model(s)" in extra.note

    # Its own phase, deliberately outside the unavoidable floor: that floor is
    # what must be spent before anything can be scored at all, and inflating
    # it turned a cap that used to buy a partial sweep into an outright
    # decline. An over-estimate that blocks a run is not the safe direction.
    assert four.unavoidable_requests == one.unavoidable_requests
    assert four.unavoidable_tokens == one.unavoidable_tokens


def test_a_capability_verdict_is_not_shared_between_unlike_models() -> None:
    """The unit the whole finding rests on: two models, two answers.

    `detect_tool_calling` offers a tool; a target that rejects the offer must
    not inherit SUPPORTED from one that accepts it.
    """
    from sweepeval.capabilities.detect import CapabilityReport, CapabilityResult

    accepts = CapabilityReport()
    accepts.results[Capability.TOOL_CALLING] = CapabilityResult(
        Capability.TOOL_CALLING, Support.SUPPORTED, "tool probe", "high"
    )
    rejects = CapabilityReport()
    rejects.results[Capability.TOOL_CALLING] = CapabilityResult(
        Capability.TOOL_CALLING, Support.INCONCLUSIVE, "tool probe", "low",
        {"status": 404},
    )
    assert accepts.supports(Capability.TOOL_CALLING)
    assert not rejects.supports(Capability.TOOL_CALLING)
    assert "INCONCLUSIVE" in rejects.skip_reason(Capability.TOOL_CALLING)


async def test_detection_happens_once_per_model_with_that_model_pinned(
    monkeypatch,
) -> None:
    """Asserted on the calls, not on the reports.

    "Every config carries a report" holds just as well when they all carry the
    *same* report -- which was the defect. Only recording what was probed, and
    with which model pinned, tells a per-model detection from a shared one.
    """
    import sweepeval.execute.sweep as sweep_module

    real = sweep_module.detect_all
    pinned_at_call: list[str | None] = []

    async def spy(client, ladder, key, text_path, **kw):
        pinned_at_call.append((ladder.pinned or {}).get("model"))
        return await real(client, ladder, key, text_path, **kw)

    monkeypatch.setattr(sweep_module, "detect_all", spy)
    await _sweep(
        "demands_a_model", cap=2, models=["gpt-mock-large", "gpt-mock-small"]
    )

    # One run-level probe with nothing pinned, then one per model.
    assert pinned_at_call[0] is None, pinned_at_call
    assert set(pinned_at_call[1:]) == {"gpt-mock-large", "gpt-mock-small"}, (
        pinned_at_call
    )


async def test_a_model_without_a_capability_runs_fewer_probes(monkeypatch) -> None:
    """The consequence that matters: the unit set follows the model.

    On the live four-model run the whole retrieval family was skipped for
    `gpt-5-search-api` -- the one model that had retrieval -- because the
    verdict came from a different model. Here one model is given multi-turn
    and the other is not, and the context family must follow.
    """
    import sweepeval.execute.sweep as sweep_module
    from sweepeval.capabilities.detect import CapabilityResult

    real = sweep_module.detect_all

    async def spy(client, ladder, key, text_path, **kw):
        report = await real(client, ladder, key, text_path, **kw)
        if (ladder.pinned or {}).get("model") == "gpt-mock-small":
            report.results[Capability.MULTI_TURN] = CapabilityResult(
                Capability.MULTI_TURN, Support.UNSUPPORTED, "spy", "high"
            )
        return report

    monkeypatch.setattr(sweep_module, "detect_all", spy)
    result = await _sweep(
        "demands_a_model", cap=2, models=["gpt-mock-large", "gpt-mock-small"]
    )

    by_model = {
        row.config.params.get("model"): {
            o.family for o in row.observations
        }
        for row in result.configs
    }
    assert "context" in by_model["gpt-mock-large"], by_model
    assert "context" not in by_model["gpt-mock-small"], by_model


async def test_probing_a_model_leaves_the_ladder_as_it_found_it() -> None:
    """The pin is scoped to the probe, including when the probe raises.

    A leaked pin is harmless today only by accident: every config's own params
    override it when the request body is built, so nothing downstream notices.
    That is a landmine rather than a safeguard -- the next reader of
    `ladder.pinned` inherits whichever model happened to be probed last.
    """
    import contextlib
    from types import SimpleNamespace

    import sweepeval.execute.sweep as sweep_module
    from sweepeval.capabilities.detect import CapabilityReport

    ladder = SimpleNamespace(pinned={"model": "original"})
    config = SimpleNamespace(params={"model": "probed"})
    base = CapabilityReport()

    async def ok(client, lad, key, text_path, **kw):
        assert lad.pinned["model"] == "probed"
        return CapabilityReport()

    async def boom(client, lad, key, text_path, **kw):
        raise RuntimeError("probe failed")

    original = sweep_module.detect_all
    try:
        sweep_module.detect_all = ok
        await sweep_module._capabilities_for(
            None, ladder, None, "$.x", "quick", config, base
        )
        assert ladder.pinned == {"model": "original"}

        sweep_module.detect_all = boom
        with contextlib.suppress(RuntimeError):
            await sweep_module._capabilities_for(
                None, ladder, None, "$.x", "quick", config, base
            )
        assert ladder.pinned == {"model": "original"}
    finally:
        sweep_module.detect_all = original
