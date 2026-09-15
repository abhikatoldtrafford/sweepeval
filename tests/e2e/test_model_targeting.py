"""Which model was measured, and the gate refusing to compare two (§16).

`baseline` and `gate` could not target a model and did not record one. The
model that answered was whatever `_preferred_model` picked out of `/v1/models`
during discovery -- first id containing `mini`, `flash`, `haiku`, `small`,
`lite` or `turbo` -- and nothing wrote it down: the manifest's target block
names the url, the shape, the auth and the extraction path, `calls.jsonl`
carries a `params_hash` and no model string, and `baseline.json` carried
`config_id: "default"`.

So a committed baseline did not say what produced it, and the gate compared
two different models without a word. Measured against api.openai.com: a
gpt-5-mini baseline gated against gpt-5-nano exited 1 on `cost_per_probe`
(1881 -> 3074, p=0.0005) and never mentioned the model.

The model is deliberately **not** a comparability key and cannot become one: a
sweep varies it across configs inside one run, and the manifest carries one
comparability block for the whole run, so a per-run key would have to lie for
every sweep. It is a property of the single config a baseline and a gate each
measure, checked where that pairing lives.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from tests.conftest import make_app, make_client
from typer.testing import CliRunner

from sweepeval.cli.main import app as cli
from sweepeval.execute.evaluate import ModelPinRefused, aevaluate_target
from sweepeval.execute.gate import (
    gate,
    gate_payload,
    load_baseline,
    model_to_pin,
    save_baseline,
    snapshot,
)
from sweepeval.schema.baseline import Baseline
from sweepeval.stats.diff import ExitCode

runner = CliRunner()


class _Recording(httpx.ASGITransport):
    """Every request body, as it went on the wire.

    The store keeps a `body_sha256` and not the body, so "the field was set"
    and "the request said so" cannot be told apart from the run directory.
    """

    def __init__(self, app: object) -> None:
        super().__init__(app=app)  # type: ignore[arg-type]
        self.bodies: list[bytes] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(request.content)
        return await super().handle_async_request(request)

    def chat_bodies(self) -> list[dict]:
        out = []
        for raw in self.bodies:
            try:
                body = json.loads(raw)
            except ValueError:
                continue
            if isinstance(body, dict) and "messages" in body:
                out.append(body)
        return out


async def _evaluate(scenario: str, tmp_path: Path, **kw):
    app = make_app(scenario)
    client = make_client(app)
    try:
        return await aevaluate_target(
            "https://mock.test" + app.scenario.paths[0],
            client=client, root=str(tmp_path), runs=2, profile="standard",
            authorized=True, authorization_prompt=False, seed=7, **kw,
        )
    finally:
        await client.aclose()


# --- what answered is written down ------------------------------------------


async def test_a_run_records_the_model_discovery_chose(tmp_path: Path) -> None:
    """`demands_a_model` exposes gpt-mock-large and gpt-mock-small, and the
    preference heuristic takes the one matching `small`. The point is not which
    it picks -- it is that the choice is now visible."""
    result = await _evaluate(
        "demands_a_model", tmp_path, key="test-key-abcdefgh"
    )
    assert result.model == "gpt-mock-small"


async def test_a_pin_is_what_the_requests_name(tmp_path: Path) -> None:
    result = await _evaluate(
        "demands_a_model", tmp_path, key="test-key-abcdefgh",
        model="gpt-mock-large",
    )
    assert result.model == "gpt-mock-large"


async def test_the_pin_reaches_the_wire_not_just_the_field(tmp_path: Path) -> None:
    """`result.model` could be set from the argument and never sent. Read it
    back off the bodies the transport carried.

    This is the one that matters: discovery reaches an endpoint like this as
    `openai.chat_completions + add:model`, and the added value has to lose to
    the pin. Get the merge order wrong and every probe still names the model
    discovery picked while the baseline records the one you asked for.
    """
    app = make_app("demands_a_model")
    transport = _Recording(app)
    client = httpx.AsyncClient(transport=transport, base_url="https://mock.test")
    try:
        result = await aevaluate_target(
            "https://mock.test" + app.scenario.paths[0],
            key="test-key-abcdefgh", client=client, root=str(tmp_path), runs=2,
            profile="standard", authorized=True, authorization_prompt=False,
            seed=7, model="gpt-mock-large",
        )
    finally:
        await client.aclose()

    assert result.model == "gpt-mock-large"

    # Discovery runs before a pin can exist and legitimately names whatever it
    # sniffed; everything from the moment the pin is set must name the pin.
    chat = transport.chat_bodies()
    first = next(
        i for i, body in enumerate(chat) if body.get("model") == "gpt-mock-large"
    )
    assert first <= 25, first  # discovery's hard budget
    after = [b.get("model") for b in chat[first:]]
    assert set(after) == {"gpt-mock-large"}, set(after)
    # Not one request: capability detection and the whole probe set.
    assert len(after) > 20, len(after)


async def test_capability_detection_runs_against_the_pinned_model(
    tmp_path: Path, monkeypatch
) -> None:
    """Otherwise the capability report describes one model and the metrics
    come from another -- and they disagree exactly where it matters, since a
    reasoning model rejects the sampling parameters a chat model accepts.

    The pin lives on the ladder rather than on the probe plan for this reason:
    five detectors and the sampling probe build their bodies straight off it.
    """
    import sweepeval.execute.evaluate as ev

    real = ev.detect_all
    seen: dict[str, dict] = {}

    async def spy(client, ladder, *args, **kwargs):
        seen["pinned"] = dict(ladder.pinned)
        return await real(client, ladder, *args, **kwargs)

    # The module that holds the reference, not the one that defines it:
    # `evaluate` imported the name, so patching the definition site patches
    # nothing this run will call.
    monkeypatch.setattr(ev, "detect_all", spy)

    await _evaluate(
        "demands_a_model", tmp_path, key="test-key-abcdefgh",
        model="gpt-mock-large",
    )
    assert seen["pinned"] == {"model": "gpt-mock-large"}


async def test_a_shape_that_carries_no_model_records_none(tmp_path: Path) -> None:
    """`None` is an answer: `openai_clean` accepts a bare messages body, so
    discovery never adds a model and the requests name none. Claiming one
    would be provenance that is not true."""
    result = await _evaluate("openai_clean", tmp_path, key="test-key-abcdefgh")
    assert result.model is None


async def test_a_pin_that_cannot_land_is_refused_not_ignored(tmp_path: Path) -> None:
    """Gemini names the model in the URL. Accepting `--model` there would put
    a model id in a committed baseline that no request ever named."""
    with pytest.raises(ModelPinRefused) as caught:
        await _evaluate(
            "gemini_shape", tmp_path, key="test-key-abcdefgh",
            model="gemini-1.5-pro",
        )
    assert "gemini" in str(caught.value)


async def test_a_refused_pin_costs_nothing_beyond_discovery(tmp_path: Path) -> None:
    """The check is the first thing after discovery, before capability
    detection and before any probe: §3 makes cost a first-class constraint,
    and the gate's own profile refusal was moved earlier for the same reason
    after it cost 120 live requests."""
    app = make_app("gemini_shape")
    transport = _Recording(app)
    client = httpx.AsyncClient(transport=transport, base_url="https://mock.test")
    try:
        with pytest.raises(ModelPinRefused):
            await aevaluate_target(
                "https://mock.test" + app.scenario.paths[0],
                key="test-key-abcdefgh", client=client, root=str(tmp_path),
                runs=2, profile="standard", authorized=True,
                authorization_prompt=False, seed=7, model="gemini-1.5-pro",
            )
    finally:
        await client.aclose()

    # Discovery's hard budget is 25. A capability sweep plus a `standard`
    # probe set is an order of magnitude more than that.
    assert len(transport.bodies) <= 25, len(transport.bodies)
    assert not list(Path(tmp_path).glob("runs/*/observations.jsonl"))


async def test_the_model_is_not_a_comparability_key(tmp_path: Path) -> None:
    """A sweep varies the model inside one run, so a per-run key would lie."""
    result = await _evaluate(
        "demands_a_model", tmp_path, key="test-key-abcdefgh"
    )
    dumped = result.comparability.model_dump(mode="json")
    assert "model" not in dumped["hard"]
    assert "model" not in dumped["soft"]


def test_a_model_never_goes_where_the_target_does_not_read_it() -> None:
    """`model` is a sampling key for the shapes that carry it in the body.
    Gemini names it in the URL and puts the rest of the sampling keys under
    `generationConfig`, so a pinned -- or swept -- model was landing in
    `generationConfig.model`: a field the API does not read and rejects.

    The refusal above does not cover this. A sweep sets the model per config
    and never goes through the pin check at all.
    """
    from sweepeval.discovery.shapes.builtin import GeminiGenerateContent

    body = GeminiGenerateContent().build_multi_turn(
        [("user", "hi")], model="gemini-1.5-pro", temperature=0.0
    )
    assert "model" not in json.dumps(body), body
    # and the sampling keys that *are* body parameters still arrive
    assert body["generationConfig"] == {"temperature": 0.0}


# --- the baseline carries it ------------------------------------------------


async def test_the_baseline_records_the_model_and_round_trips(tmp_path: Path) -> None:
    result = await _evaluate(
        "demands_a_model", tmp_path, key="test-key-abcdefgh"
    )
    path = save_baseline(snapshot(result), tmp_path / "baseline.json")
    assert json.loads(path.read_text(encoding="utf-8"))["model"] == "gpt-mock-small"
    assert load_baseline(path).model == "gpt-mock-small"


def test_a_baseline_written_before_this_field_still_loads(tmp_path: Path) -> None:
    """I6 back-compat: a committed baseline must not stop working because a
    field was added to the format."""
    payload = json.loads(_MINIMAL_BASELINE)
    path = tmp_path / "old.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_baseline(path).model is None


# --- and the gate refuses to compare two of them ----------------------------


async def test_the_gate_refuses_a_different_model(tmp_path: Path) -> None:
    baseline = snapshot(
        await _evaluate("demands_a_model", tmp_path / "a", key="test-key-abcdefgh")
    )
    current = await _evaluate(
        "demands_a_model", tmp_path / "b", key="test-key-abcdefgh",
        model="gpt-mock-large",
    )

    verdict = gate(current, baseline, seed=1)
    assert verdict.exit_code is ExitCode.COMPARABILITY_REFUSED
    assert not verdict.ok
    assert any("gpt-mock-small" in r and "gpt-mock-large" in r for r in verdict.refusals)


async def test_the_refusal_says_it_is_not_a_regression(tmp_path: Path) -> None:
    """Exit 1 and exit 2 are different facts. A model swap reported as a
    regression is how a gate teaches people to ignore it."""
    baseline = snapshot(
        await _evaluate("demands_a_model", tmp_path / "a", key="test-key-abcdefgh")
    )
    current = await _evaluate(
        "demands_a_model", tmp_path / "b", key="test-key-abcdefgh",
        model="gpt-mock-large",
    )
    verdict = gate(current, baseline, seed=1)
    assert not verdict.regressions
    assert any("re-baseline" in n for n in verdict.notes)


async def test_the_same_model_gates_normally(tmp_path: Path) -> None:
    """The check must not refuse the case it exists to protect."""
    baseline = snapshot(
        await _evaluate("demands_a_model", tmp_path / "a", key="test-key-abcdefgh")
    )
    current = await _evaluate(
        "demands_a_model", tmp_path / "b", key="test-key-abcdefgh"
    )
    verdict = gate(current, baseline, seed=1)
    assert verdict.exit_code is not ExitCode.COMPARABILITY_REFUSED
    assert not any("model differs" in r for r in verdict.refusals)


async def test_allow_model_change_compares_and_says_so(tmp_path: Path) -> None:
    baseline = snapshot(
        await _evaluate("demands_a_model", tmp_path / "a", key="test-key-abcdefgh")
    )
    current = await _evaluate(
        "demands_a_model", tmp_path / "b", key="test-key-abcdefgh",
        model="gpt-mock-large",
    )
    verdict = gate(current, baseline, seed=1, allow_model_change=True)

    assert verdict.exit_code is not ExitCode.COMPARABILITY_REFUSED
    assert any("allow-model-change" in n for n in verdict.notes), verdict.notes
    # Annotated on the verdict the statistics produced, not instead of it.
    assert verdict.diffs


async def test_an_unrecorded_baseline_is_annotated_not_refused(tmp_path: Path) -> None:
    """The gate cannot check an older baseline. Treating "unknown" as
    "matching" without saying so is exactly what this check exists to stop."""
    result = await _evaluate(
        "demands_a_model", tmp_path, key="test-key-abcdefgh"
    )
    baseline = snapshot(result).model_copy(update={"model": None})

    verdict = gate(result, baseline, seed=1)
    assert verdict.exit_code is not ExitCode.COMPARABILITY_REFUSED
    assert any("predates model recording" in n for n in verdict.notes), verdict.notes


async def test_both_models_reach_the_payload_even_on_a_refusal(tmp_path: Path) -> None:
    baseline = snapshot(
        await _evaluate("demands_a_model", tmp_path / "a", key="test-key-abcdefgh")
    )
    current = await _evaluate(
        "demands_a_model", tmp_path / "b", key="test-key-abcdefgh",
        model="gpt-mock-large",
    )
    payload = gate_payload(gate(current, baseline, seed=1))
    assert payload["exit_code"] == int(ExitCode.COMPARABILITY_REFUSED)
    assert payload["model"] == {
        "baseline": "gpt-mock-small", "current": "gpt-mock-large",
    }


# --- the gate re-measures the baseline's model by default -------------------


def test_the_pin_defaults_to_the_baselines_model() -> None:
    baseline = Baseline.model_validate_json(_MINIMAL_BASELINE).model_copy(
        update={"model": "gpt-mock-small"}
    )
    assert model_to_pin(baseline, None) == "gpt-mock-small"


def test_an_explicit_pin_wins() -> None:
    baseline = Baseline.model_validate_json(_MINIMAL_BASELINE).model_copy(
        update={"model": "gpt-mock-small"}
    )
    assert model_to_pin(baseline, "gpt-mock-large") == "gpt-mock-large"


def test_allowing_a_change_stops_pinning() -> None:
    """Otherwise --allow-model-change would pin the baseline's model and then
    report that nothing changed."""
    baseline = Baseline.model_validate_json(_MINIMAL_BASELINE).model_copy(
        update={"model": "gpt-mock-small"}
    )
    assert model_to_pin(baseline, None, allow_model_change=True) is None


def test_an_unrecorded_baseline_pins_nothing() -> None:
    baseline = Baseline.model_validate_json(_MINIMAL_BASELINE)
    assert model_to_pin(baseline, None) is None


def test_both_re_run_paths_share_the_default() -> None:
    """The CLI and `api.arun_gate` are the two ways to re-run and gate, and
    they split their arguments by hand. A default implemented in one of them
    is a default the other silently does not have."""
    import inspect

    from sweepeval import api
    from sweepeval.cli import evaluate as cli_evaluate

    assert "model_to_pin" in inspect.getsource(api.arun_gate)
    assert "model_to_pin" in inspect.getsource(cli_evaluate.gate_command)


def test_allow_model_change_is_a_gate_argument_not_a_run_one() -> None:
    """`arun_gate` forwards everything it does not recognise to `aevaluate`,
    where this would be an unexpected keyword."""
    import inspect

    from sweepeval import api

    source = inspect.getsource(api.arun_gate)
    forwarded = source.split("gate_kwargs = {")[1].split("}")[0]
    assert "allow_model_change" in forwarded


# --- the CLI surface --------------------------------------------------------


def test_every_verb_that_runs_a_target_can_pin_it() -> None:
    from tests.conftest import cli_help

    for verb in ("evaluate", "baseline", "gate"):
        assert "--model" in cli_help(verb), verb
    assert "--allow-model-change" in cli_help("gate")


def test_the_cli_flag_reaches_the_comparison(tmp_path: Path, monkeypatch) -> None:
    """A flag that is parsed, documented and never forwarded is the shape of
    defect this whole change exists to remove."""
    import sweepeval.cli.evaluate as cli_evaluate

    path = tmp_path / "baseline.json"
    path.write_text(_MINIMAL_BASELINE, encoding="utf-8")

    seen: dict[str, object] = {}

    def fake_run(*_args, **_kwargs):
        return object()

    def fake_gate(_result, _baseline, **kwargs):
        seen.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(cli_evaluate, "_run", fake_run)
    monkeypatch.setattr(cli_evaluate, "gate_result", fake_gate)
    runner.invoke(
        cli,
        ["gate", "https://mock.test/v1/chat/completions", "--baseline", str(path),
         "--allow-model-change", "--yes"],
    )
    assert seen.get("allow_model_change") is True, seen


def test_a_refused_pin_is_a_usage_error(tmp_path: Path, monkeypatch) -> None:
    """Exit 3, not 2: the flag is wrong for this target, which is a fact about
    the command line, not about two results being incomparable."""
    import sweepeval.cli.evaluate as cli_evaluate

    async def refuse(*_args, **_kwargs):
        raise ModelPinRefused("nope")

    monkeypatch.setattr(cli_evaluate, "aevaluate_target", refuse)
    result = runner.invoke(
        cli, ["evaluate", "https://mock.test/v1/chat/completions",
              "--model", "x", "--yes"]
    )
    assert result.exit_code == int(ExitCode.USAGE_ERROR), result.output


# A baseline with the minimum the schema requires and no `model` field, which
# is what a file committed before this release looks like.
_MINIMAL_BASELINE = json.dumps(
    {
        "run_id": "r",
        "created_at": "2026-01-01T00:00:00+00:00",
        "config_id": "default",
        "comparability": {
            "hard": {
                "schema_major": 1,
                "suite_version": 1,
                "corpus_hash": "h",
                "probe_layers": ["generic"],
                "target_type": "BARE_MODEL",
                "similarity_backend": "lexical",
                "judge": None,
                "profile": "standard",
                "pricing_source": "none",
                "scorer_versions": {},
                "extraction_path": "$.x",
            },
            "soft": {"n_runs": 3, "concurrency": 1, "tool_version": "0"},
        },
        "metrics": {},
        "clusters": {},
    }
)
